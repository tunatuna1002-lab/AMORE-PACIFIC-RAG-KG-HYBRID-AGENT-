"""Prompt management module"""
from pathlib import Path
from typing import Dict

class PromptLoader:
    """Load and cache prompt templates"""

    _cache: Dict[str, str] = {}
    _prompts_dir = Path(__file__).parent

    @classmethod
    def get(cls, name: str) -> str:
        """Get prompt by name (without .txt extension)"""
        if name not in cls._cache:
            prompt_file = cls._prompts_dir / f"{name}.txt"
            if prompt_file.exists():
                cls._cache[name] = prompt_file.read_text(encoding='utf-8')
            else:
                raise FileNotFoundError(f"Prompt not found: {name}")
        return cls._cache[name]

    @classmethod
    def format(cls, name: str, **kwargs) -> str:
        """Get prompt and format with variables"""
        template = cls.get(name)
        return template.format(**kwargs)
