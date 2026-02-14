"""LLM Client Protocol definition"""

from typing import Any, Protocol


class LLMClientProtocol(Protocol):
    """Protocol for LLM client implementations"""

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str: ...

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str: ...

    async def generate_with_context(
        self,
        prompt: str,
        context: list[dict[str, Any]],
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str: ...
