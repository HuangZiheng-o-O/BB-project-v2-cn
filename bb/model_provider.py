"""统一的模型端口:提供兼容 Anthropic 与 OpenAI 协议的适配器。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse


@dataclass(frozen=True)
class ToolCall:
    """模型发起的一次工具调用。"""

    call_id: str  # 调用 ID,工具结果需要用它回传给模型
    name: str  # 工具名
    arguments: dict[str, Any]  # 工具参数


@dataclass(frozen=True)
class ModelTurn:
    """模型的一轮输出。"""

    text: str  # 文本内容
    tool_calls: list[ToolCall] = field(default_factory=list)  # 本轮发起的工具调用
    usage: dict[str, int] = field(default_factory=dict)  # token 用量
    stop_reason: str = ""  # 停止原因
    response_items: list[Any] = field(default_factory=list)  # 原始响应条目(OpenAI Responses API 续写时需要)


class ModelPort(Protocol):
    """模型端口协议:任何实现了 model_name 与 generate 的对象都可以充当模型。"""

    model_name: str

    def generate(
        self,
        system: str,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 5000,
        json_mode: bool = False,
    ) -> ModelTurn: ...


class AnthropicCompatibleModel:
    """兼容 Anthropic Messages 协议的模型适配器。"""

    def __init__(self, model_name: str, base_url: str | None = None) -> None:
        # 延迟导入,未使用该提供方时不要求安装依赖
        from anthropic import Anthropic

        # 认证:ANTHROPIC_AUTH_TOKEN 或 ANTHROPIC_API_KEY 至少设置一个
        token = os.getenv("ANTHROPIC_AUTH_TOKEN")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not token and not api_key:
            raise RuntimeError("Set ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY")
        self.model_name = model_name
        self.client = Anthropic(
            auth_token=token or None,
            api_key=api_key or None,
            base_url=base_url or os.getenv("ANTHROPIC_BASE_URL"),
            # 超时时间可通过环境变量配置,默认 180 秒
            timeout=float(os.getenv("BB_MODEL_TIMEOUT_SECONDS", "180")),
            max_retries=2,
        )

    @staticmethod
    def _messages(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """把内部统一格式的对话历史转换成 Anthropic 的 messages 结构。"""
        messages: list[dict[str, Any]] = []
        for turn in history:
            if turn["role"] == "user":
                messages.append({"role": "user", "content": turn["content"]})
            elif turn["role"] == "assistant":
                blocks: list[dict[str, Any]] = []
                if turn.get("content"):
                    blocks.append({"type": "text", "text": turn["content"]})
                # 助手发起的工具调用转成 tool_use 块
                for call in turn.get("tool_calls", []):
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call["call_id"],
                            "name": call["name"],
                            "input": call["arguments"],
                        }
                    )
                messages.append({"role": "assistant", "content": blocks})
            elif turn["role"] == "tool":
                # 工具结果转成 tool_result 块
                block = {
                    "type": "tool_result",
                    "tool_use_id": turn["tool_call_id"],
                    "content": turn["content"],
                }
                # 同一轮的多个工具结果要合并进同一条 user 消息里
                if messages and messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list):
                    messages[-1]["content"].append(block)
                else:
                    messages.append({"role": "user", "content": [block]})
        return messages

    def generate(
        self,
        system: str,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 5000,
        json_mode: bool = False,
    ) -> ModelTurn:
        """调用模型生成一轮输出。json_mode 在此适配器中不使用(靠提示词约束 JSON)。"""
        args: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "system": system,
            "messages": self._messages(history),
        }
        if tools:
            # 内部的 parameters 对应 Anthropic 的 input_schema
            args["tools"] = [
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["parameters"],
                }
                for tool in tools
            ]
        response = self.client.messages.create(**args)
        # 多个文本块用换行拼接
        text = "\n".join(block.text for block in response.content if block.type == "text")
        calls = [
            ToolCall(call_id=block.id, name=block.name, arguments=block.input)
            for block in response.content
            if block.type == "tool_use"
        ]
        usage = {
            "input_tokens": int(response.usage.input_tokens or 0),
            "output_tokens": int(response.usage.output_tokens or 0),
        }
        return ModelTurn(text=text, tool_calls=calls, usage=usage, stop_reason=response.stop_reason or "")


class OpenAICompatibleModel:
    """兼容 OpenAI 协议的模型适配器(也可用于智谱 z.ai 等兼容服务)。"""

    def __init__(self, model_name: str, base_url: str | None = None) -> None:
        # 延迟导入,未使用该提供方时不要求安装依赖
        from openai import OpenAI

        endpoint = base_url or os.getenv("OPENAI_BASE_URL")
        # 没有显式端点、也没有 OpenAI 密钥时,回退到 ZAI_BASE_URL
        if not endpoint and not os.getenv("OPENAI_API_KEY"):
            endpoint = os.getenv("ZAI_BASE_URL")
        # 端点主机是 api.z.ai 时使用 ZAI_API_KEY,否则使用 OPENAI_API_KEY
        key_name = "ZAI_API_KEY" if endpoint and urlparse(endpoint).hostname == "api.z.ai" else "OPENAI_API_KEY"
        key = os.getenv(key_name)
        if not key:
            raise RuntimeError(f"Set {key_name}")
        self.model_name = model_name
        self.client = OpenAI(
            api_key=key,
            base_url=endpoint,
            timeout=float(os.getenv("BB_MODEL_TIMEOUT_SECONDS", "180")),
            max_retries=2,
        )

    def generate(
        self,
        system: str,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 5000,
        json_mode: bool = False,
    ) -> ModelTurn:
        """调用模型生成一轮输出;官方 OpenAI 的 gpt-6-* 模型走 Responses API,其余走 Chat Completions。"""
        if self.model_name.startswith("gpt-6-") and urlparse(str(self.client.base_url)).hostname == "api.openai.com":
            return self._generate_responses(system, history, tools, max_tokens, json_mode)
        # 把内部对话历史转换成 Chat Completions 的 messages
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for turn in history:
            if turn["role"] in ("user", "assistant"):
                payload: dict[str, Any] = {"role": turn["role"], "content": turn.get("content") or ""}
                if turn["role"] == "assistant" and turn.get("tool_calls"):
                    # 助手的工具调用转成 OpenAI 的 function tool_calls 格式(参数需序列化为字符串)
                    payload["tool_calls"] = [
                        {
                            "id": call["call_id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call["arguments"]),
                            },
                        }
                        for call in turn["tool_calls"]
                    ]
                messages.append(payload)
            elif turn["role"] == "tool":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": turn["tool_call_id"],
                        "content": turn["content"],
                    }
                )
        args: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            args["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    },
                }
                for tool in tools
            ]
        if json_mode:
            # 要求服务端只返回合法 JSON 对象
            args["response_format"] = {"type": "json_object"}
        # GLM 系列模型的"思考模式"开关:可由 BB_GLM_THINKING 指定;
        # 未指定时,glm-4.7* 默认关闭思考
        thinking_mode = os.getenv("BB_GLM_THINKING")
        if not thinking_mode and self.model_name.lower().startswith("glm-4.7"):
            thinking_mode = "disabled"
        if thinking_mode:
            if thinking_mode not in {"enabled", "disabled"}:
                raise ValueError("BB_GLM_THINKING must be enabled or disabled")
            args["extra_body"] = {"thinking": {"type": thinking_mode}}
        response = self.client.chat.completions.create(**args)
        message = response.choices[0].message
        # 工具参数以 JSON 字符串形式返回,需要解析
        calls = [
            ToolCall(
                call_id=call.id,
                name=call.function.name,
                arguments=json.loads(call.function.arguments),
            )
            for call in (message.tool_calls or [])
        ]
        usage = response.usage
        return ModelTurn(
            text=message.content or "",
            tool_calls=calls,
            usage={
                "input_tokens": int(usage.prompt_tokens) if usage else 0,
                "output_tokens": int(usage.completion_tokens) if usage else 0,
            },
            stop_reason=response.choices[0].finish_reason or "",
        )

    def _generate_responses(
        self,
        system: str,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        json_mode: bool,
    ) -> ModelTurn:
        """使用 OpenAI Responses API 生成一轮输出。"""
        inputs: list[Any] = []
        for turn in history:
            if turn["role"] == "assistant" and turn.get("response_items"):
                # 有原始响应条目时原样回放(保留推理的加密内容,以便模型续写)
                inputs.extend(turn["response_items"])
            elif turn["role"] in ("user", "assistant"):
                if turn.get("content"):
                    inputs.append({"role": turn["role"], "content": turn["content"]})
                for call in turn.get("tool_calls", []):
                    inputs.append({
                        "type": "function_call",
                        "call_id": call["call_id"],
                        "name": call["name"],
                        "arguments": json.dumps(call["arguments"]),
                    })
            elif turn["role"] == "tool":
                inputs.append({
                    "type": "function_call_output",
                    "call_id": turn["tool_call_id"],
                    "output": turn["content"],
                })
        args: dict[str, Any] = {
            "model": self.model_name,
            "instructions": system,
            "input": inputs,
            # 给推理过程额外预留 2000 个输出 token
            "max_output_tokens": max_tokens + 2000,
            # 推理强度可通过环境变量配置,默认 medium
            "reasoning": {"effort": os.getenv("BB_OPENAI_REASONING_EFFORT", "medium")},
            "store": False,  # 不在服务端保存对话
            "include": ["reasoning.encrypted_content"],
        }
        if tools:
            args["tools"] = [
                {
                    "type": "function",
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                    "strict": False,
                }
                for tool in tools
            ]
        if json_mode:
            # Responses API 的 JSON 模式要求输入里明确提到 JSON,这里追加一句提示
            inputs.append({"role": "user", "content": "Return one valid JSON object."})
            args["text"] = {"format": {"type": "json_object"}}
        response = self.client.responses.create(**args)
        calls = [
            ToolCall(call_id=item.call_id, name=item.name, arguments=json.loads(item.arguments))
            for item in response.output if item.type == "function_call"
        ]
        return ModelTurn(
            text=response.output_text,
            tool_calls=calls,
            usage={
                "input_tokens": int(response.usage.input_tokens) if response.usage else 0,
                "output_tokens": int(response.usage.output_tokens) if response.usage else 0,
            },
            stop_reason=response.status or "",
            response_items=list(response.output),
        )


def make_model(provider: str, model_name: str, base_url: str | None = None) -> ModelPort:
    """工厂函数:按提供方名称创建对应的模型适配器。"""
    if provider == "anthropic":
        return AnthropicCompatibleModel(model_name, base_url)
    if provider == "openai":
        return OpenAICompatibleModel(model_name, base_url)
    raise ValueError(f"Unsupported provider: {provider}")


def parse_json_object(text: str) -> dict[str, Any]:
    """从模型输出中提取一个 JSON 对象;对象之外多余的文字不会被当作数据接受。"""
    stripped = text.strip()
    # 去掉 Markdown 代码围栏(```json ... ```)
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    # 不以 { 开头时,截取第一个 { 到最后一个 } 之间的内容
    if not stripped.startswith("{"):
        first = stripped.find("{")
        last = stripped.rfind("}")
        if first < 0 or last <= first:
            raise ValueError("Model did not return a JSON object")
        stripped = stripped[first : last + 1]
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def generate_json(
    model: ModelPort,
    system: str,
    user: str,
    max_tokens: int,
    attempts: int = 2,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """调用模型并解析出 JSON 对象;解析失败则把错误反馈给模型重试,最多 attempts 次。"""
    history: list[dict[str, Any]] = [{"role": "user", "content": user}]
    trace: list[dict[str, Any]] = []
    for attempt in range(attempts):
        turn = model.generate(system, history, max_tokens=max_tokens, json_mode=True)
        trace.append({"stage": "json", "attempt": attempt + 1, "usage": turn.usage, "stop_reason": turn.stop_reason})
        try:
            return parse_json_object(turn.text), trace
        except (ValueError, json.JSONDecodeError) as error:
            # 把上一次的输出和错误原因追加进历史,让模型修正
            history.extend(
                [
                    {"role": "assistant", "content": turn.text, "response_items": turn.response_items},
                    {"role": "user", "content": f"Your output was not valid JSON: {error}. Return one complete JSON object only."},
                ]
            )
    raise ValueError("Model failed to return valid JSON after retry")
