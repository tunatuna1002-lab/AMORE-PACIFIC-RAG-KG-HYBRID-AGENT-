"""LLM Client Protocol definition"""
from typing import Protocol, List, Dict, Any, Optional

class LLMClientProtocol(Protocol):
    """Protocol for LLM client implementations"""
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str: ...

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str: ...
