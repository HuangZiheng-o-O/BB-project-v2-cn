"""只读的来源文档导入,以及可按行定位的词法检索。"""

from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable

from bb.models import Anchor


@dataclass(frozen=True)
class Source:
    """一份来源文档(不可变):ID、文件名、绝对路径、内容哈希和按行切分的文本。"""

    source_id: str
    filename: str
    absolute_path: str
    sha256: str
    lines: tuple[str, ...]

    def formatted(self, first: int = 1, last: int | None = None) -> str:
        """输出带行号的文本,每行形如 "L0012 内容";first/last 为 1 起始的闭区间。"""
        last = min(last or len(self.lines), len(self.lines))
        body = "\n".join(
            f"L{number:04d} {self.lines[number - 1]}"
            for number in range(max(first, 1), last + 1)
        )
        return f"DOCUMENT {self.source_id} ({self.filename})\n{body}"


class Corpus:
    """稳定的"检索/来源"端口的小型实现。

    将来可以用别的检索适配器替换 FTS5,而无需改变来源锚点。
    """

    def __init__(self, documents: Path, index_path: Path) -> None:
        # 文档根目录必须真实存在(strict=True),并解析为绝对路径
        self.documents = documents.resolve(strict=True)
        self.sources: dict[str, Source] = {}
        # 允许跨线程使用同一连接,检索时用锁串行化
        self.db = sqlite3.connect(index_path, check_same_thread=False)
        self._search_lock = threading.Lock()
        self.db.row_factory = sqlite3.Row
        # 来源元数据表
        self.db.execute(
            "CREATE TABLE source (source_id TEXT PRIMARY KEY, filename TEXT NOT NULL, "
            "absolute_path TEXT NOT NULL, sha256 TEXT NOT NULL, line_count INTEGER NOT NULL)"
        )
        # 按行建立的全文检索虚拟表(FTS5);来源 ID 与行号只存储、不参与索引
        self.db.execute(
            "CREATE VIRTUAL TABLE source_fts USING fts5(source_id UNINDEXED, "
            "line_number UNINDEXED, content, tokenize='unicode61')"
        )
        self._load()

    def _load(self) -> None:
        """递归读取文档根目录下所有 .txt 文件,登记来源并写入索引。"""
        paths = sorted(self.documents.rglob("*.txt"))
        if not paths:
            raise ValueError(f"No .txt documents found in {self.documents}")
        for path in paths:
            # 安全检查:拒绝符号链接,且文件必须位于文档根目录之内
            if path.is_symlink():
                raise ValueError(f"Symlink sources are not accepted: {path}")
            absolute = path.resolve(strict=True)
            if not absolute.is_relative_to(self.documents):
                raise ValueError(f"Source escaped document root: {path}")
            raw = absolute.read_bytes()
            # utf-8-sig 会自动去掉可能存在的 BOM
            content = raw.decode("utf-8-sig")
            lines = tuple(content.splitlines())
            # 优先使用文档内 "Document ID: xxx" 作为来源 ID,没有则用文件名(不含扩展名)
            match = re.search(r"^Document ID:\s*(\S+)", content, re.MULTILINE)
            source_id = match.group(1) if match else path.stem
            # ID 重复时追加相对路径哈希前 8 位,保证唯一
            if source_id in self.sources:
                relative_name = str(absolute.relative_to(self.documents))
                source_id = f"{source_id}-{sha256(relative_name.encode()).hexdigest()[:8]}"
            source = Source(
                source_id=source_id,
                filename=str(absolute.relative_to(self.documents)),
                absolute_path=str(absolute),
                sha256=sha256(raw).hexdigest(),
                lines=lines,
            )
            self.sources[source_id] = source
            self.db.execute(
                "INSERT INTO source VALUES (?, ?, ?, ?, ?)",
                (source_id, source.filename, source.absolute_path, source.sha256, len(lines)),
            )
            # 每一行单独一条全文索引记录,行号从 1 开始
            self.db.executemany(
                "INSERT INTO source_fts (source_id, line_number, content) VALUES (?, ?, ?)",
                ((source_id, number, line) for number, line in enumerate(lines, 1)),
            )
        self.db.commit()

    def manifest(self) -> dict[str, str]:
        """返回 {来源ID: 内容哈希} 清单,用于判断文档是否发生变化。"""
        return {source_id: source.sha256 for source_id, source in self.sources.items()}

    def get(self, source_id: str) -> Source:
        """按 ID 取来源文档;不存在会抛 KeyError。"""
        return self.sources[source_id]

    def validate_anchor(self, anchor: Anchor) -> None:
        """校验锚点:来源必须存在,且所有行号都不超过该文档的总行数。"""
        if anchor.source_id not in self.sources:
            raise ValueError(f"Unknown source: {anchor.source_id}")
        maximum = len(self.sources[anchor.source_id].lines)
        if any(number > maximum for number in anchor.lines):
            raise ValueError(f"Line outside {anchor.source_id}: {anchor.lines}")

    def quote(self, anchor: Anchor) -> str:
        """取出锚点所指行的原文,去掉首尾空白后用空格拼成一段。"""
        self.validate_anchor(anchor)
        source = self.get(anchor.source_id)
        return " ".join(source.lines[number - 1].strip() for number in anchor.lines)

    def open(self, source_id: str, first: int = 1, last: int | None = None) -> str:
        """打开文档的指定行区间(带行号)。"""
        source = self.get(source_id)
        return source.formatted(first, last)

    def search(self, query: str, limit: int = 15, source_ids: Iterable[str] | None = None) -> list[dict]:
        """用 FTS5 做词法检索,按 BM25 相关度排序;结果不完整,不能用于穷举计数。"""
        # 提取词元(支持 Unicode),最多取前 12 个
        tokens = re.findall(r"[\w]+", query, flags=re.UNICODE)
        if not tokens:
            return []
        tokens = tokens[:12]
        # 每个词元加引号后用 OR 连接,避免特殊字符被当作检索语法
        fts_query = " OR ".join(f'"{token}"' for token in tokens)
        permitted = set(source_ids) if source_ids is not None else None
        # 先多取一些行(limit 的 8 倍,上限 500),再在 Python 侧按来源过滤。
        with self._search_lock:
            rows = self.db.execute(
                "SELECT source_id, line_number, content, bm25(source_fts) AS score "
                "FROM source_fts WHERE source_fts MATCH ? ORDER BY score LIMIT ?",
                (fts_query, min(max(limit * 8, limit), 500)),
            ).fetchall()
        results = []
        for row in rows:
            # 不在允许范围内的来源直接跳过
            if permitted is not None and row["source_id"] not in permitted:
                continue
            results.append(
                {
                    "source_id": row["source_id"],
                    "line": int(row["line_number"]),
                    "text": row["content"],
                    "score": float(row["score"]),
                }
            )
            if len(results) == limit:
                break
        return results

    def extraction_batches(self, max_chars: int = 13500) -> list[str]:
        """把所有文档切成若干批文本,每批不超过 max_chars,供模型分批抽取。"""
        batches: list[str] = []
        pending: list[str] = []
        current = 0
        for source in self.sources.values():
            parts = self._source_parts(source, max_chars)
            for part in parts:
                # 加入当前批会超限 → 先收尾当前批,再开新批
                if pending and current + len(part) > max_chars:
                    batches.append("\n\n".join(pending))
                    pending, current = [], 0
                pending.append(part)
                current += len(part)
        if pending:
            batches.append("\n\n".join(pending))
        return batches

    @staticmethod
    def _source_parts(source: Source, max_chars: int) -> list[str]:
        """把单个文档按行切成不超过 max_chars 的若干段(短文档整体作为一段)。"""
        if len(source.formatted()) <= max_chars:
            return [source.formatted()]
        parts: list[str] = []
        first = 1
        while first <= len(source.lines):
            last = first
            size = 0
            # 逐行累加长度;预留 200 字符给文档头,每行额外计 8 字符的行号前缀
            while last <= len(source.lines) and size + len(source.lines[last - 1]) < max_chars - 200:
                size += len(source.lines[last - 1]) + 8
                last += 1
            # 至少保证每段包含一行,避免单行过长时死循环
            last = max(first, last - 1)
            parts.append(source.formatted(first, last))
            first = last + 1
        return parts
