"""不可变、按内容寻址的模型阶段缓存,用于让长时间运行可重复、可恢复。"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any


class StageCache:
    """阶段缓存:以 (阶段, 模型, 提示词, 载荷) 的哈希作为文件名,把模型输出落盘。"""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        # 缓存目录不存在则自动创建(含父目录)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, stage: str, model: str, prompt: str, payload: str) -> Path:
        """根据阶段名、模型名、提示词和输入载荷计算缓存文件路径。"""
        # 任一输入变化都会得到不同的 key,从而自动让旧缓存失效
        key = json.dumps([stage, model, prompt, payload], ensure_ascii=False)
        digest = sha256(key.encode()).hexdigest()
        return self.directory / f"{stage}-{digest}.json"

    @staticmethod
    def read(path: Path) -> dict[str, Any] | None:
        """读取缓存;文件不存在、损坏或内容不是 JSON 对象时返回 None(视为未命中)。"""
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def write(path: Path, value: dict[str, Any]) -> None:
        """写入缓存。使用 "x" 独占模式:文件已存在会报错,保证缓存一经写入就不可变。"""
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False)
            stream.write("\n")
