"""
LLM Client Module
Shared LLM client to standardize API calls and eliminate duplicate patterns.

This module provides a unified interface for LLM interactions across the codebase,
handling retries, error logging, and response formatting consistently.
"""

import json
import time
from typing import Optional, Dict, Any, List
from litellm import acompletion
from src.monitoring.logger import AgentLogger


class LLMError(Exception):
    """Custom exception for LLM-related errors."""
    pass


class LLMClient:
    """
    Unified LLM client for standardized API interactions.

    This client wraps litellm.acompletion with:
    - Configurable model and temperature defaults
    - Automatic retry logic with exponential backoff
    - Comprehensive error logging
    - Token usage tracking
    - JSON response parsing

    Usage:
        # Basic completion
        client = LLMClient()
        response = await client.complete(
            system_prompt="You are a helpful assistant.",
            user_prompt="Explain quantum computing."
        )

        # JSON completion
        json_response = await client.complete_json(
            system_prompt="You are a data analyst.",
            user_prompt="Analyze this data: {...}"
        )

        # Custom model and temperature
        custom_client = LLMClient(model="gpt-4", default_temperature=0.5)
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        default_temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        logger: Optional[AgentLogger] = None
    ):
        """
        Initialize the LLM client.

        Args:
            model: Default model to use for completions (e.g., "gpt-4.1-mini", "gpt-4")
            default_temperature: Default temperature for sampling (0.0 = deterministic, 2.0 = very random)
            max_retries: Maximum number of retry attempts on failure
            retry_delay: Initial delay between retries in seconds (uses exponential backoff)
            logger: Optional logger instance for tracking API calls
        """
        self.model = model
        self.default_temperature = default_temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logger or AgentLogger("llm_client")

        # Statistics tracking
        self._total_calls = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_errors = 0

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 2000,
        model: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate a text completion from the LLM.

        Args:
            system_prompt: System-level instructions for the model
            user_prompt: User's input/question
            temperature: Sampling temperature (overrides default if provided)
            max_tokens: Maximum tokens in the response
            model: Model to use (overrides default if provided)
            messages: Optional pre-constructed messages list (overrides system/user prompts)

        Returns:
            The generated text response as a string

        Raises:
            LLMError: If all retry attempts fail or response is invalid
        """
        effective_model = model or self.model
        effective_temperature = temperature if temperature is not None else self.default_temperature

        # Construct messages
        if messages is None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

        self.logger.llm_request(effective_model, prompt_tokens=None)
        start_time = time.time()

        # Retry loop
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await acompletion(
                    model=effective_model,
                    messages=messages,
                    temperature=effective_temperature,
                    max_tokens=max_tokens
                )

                # Extract response
                if not response.choices:
                    raise LLMError("LLM returned empty choices list")

                content = response.choices[0].message.content

                # Track statistics
                latency_ms = (time.time() - start_time) * 1000
                self._total_calls += 1

                if hasattr(response, 'usage'):
                    self._total_prompt_tokens += response.usage.prompt_tokens
                    self._total_completion_tokens += response.usage.completion_tokens

                    self.logger.llm_response(
                        effective_model,
                        completion_tokens=response.usage.completion_tokens,
                        latency_ms=latency_ms
                    )
                else:
                    self.logger.llm_response(effective_model, latency_ms=latency_ms)

                return content

            except Exception as e:
                last_error = e
                self._total_errors += 1

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(
                        f"LLM call failed (attempt {attempt + 1}/{self.max_retries}), retrying in {delay}s",
                        {"error": str(e), "model": effective_model}
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(
                        f"LLM call failed after {self.max_retries} attempts",
                        {"error": str(e), "model": effective_model},
                        exc_info=True
                    )

        # All retries failed
        raise LLMError(f"Failed to complete LLM request after {self.max_retries} attempts: {last_error}")

    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = 0.3,
        max_tokens: int = 2000,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a JSON completion from the LLM.

        This method automatically instructs the model to return valid JSON
        and parses the response into a Python dictionary.

        Args:
            system_prompt: System-level instructions (JSON format will be enforced)
            user_prompt: User's input/question
            temperature: Sampling temperature (defaults to 0.3 for more deterministic JSON)
            max_tokens: Maximum tokens in the response
            model: Model to use (overrides default if provided)

        Returns:
            Parsed JSON response as a Python dictionary

        Raises:
            LLMError: If response is not valid JSON or request fails
        """
        # Enhance system prompt to enforce JSON
        json_system_prompt = f"{system_prompt}\n\nIMPORTANT: You MUST respond with valid JSON only. Do not include any text outside the JSON structure."
        json_user_prompt = f"{user_prompt}\n\nRespond with valid JSON only."

        # Get text response
        response_text = await self.complete(
            system_prompt=json_system_prompt,
            user_prompt=json_user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model
        )

        # Parse JSON
        try:
            # Remove markdown code blocks if present
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()

            return json.loads(cleaned_text)

        except json.JSONDecodeError as e:
            self.logger.error(
                "Failed to parse JSON response from LLM",
                {"error": str(e), "response_preview": response_text[:200]}
            )
            raise LLMError(f"LLM response was not valid JSON: {e}")

    async def complete_with_usage(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 2000,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a completion and return both content and usage statistics.

        Args:
            system_prompt: System-level instructions
            user_prompt: User's input/question
            temperature: Sampling temperature
            max_tokens: Maximum tokens in the response
            model: Model to use

        Returns:
            Dictionary containing:
                - content: The generated text
                - model: Model used
                - usage: Token usage statistics (if available)
                - latency_ms: Response latency in milliseconds
        """
        effective_model = model or self.model
        effective_temperature = temperature if temperature is not None else self.default_temperature

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        start_time = time.time()

        response = await acompletion(
            model=effective_model,
            messages=messages,
            temperature=effective_temperature,
            max_tokens=max_tokens
        )

        latency_ms = (time.time() - start_time) * 1000

        result = {
            "content": response.choices[0].message.content if response.choices else "",
            "model": effective_model,
            "latency_ms": latency_ms,
            "usage": None
        }

        if hasattr(response, 'usage'):
            result["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

        return result

    def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: Optional[str] = None
    ) -> float:
        """
        Estimate the cost of an LLM call based on token usage.

        Pricing (as of 2025-01):
        - gpt-4.1-mini: $0.40/1M input, $1.60/1M output
        - gpt-4: $30/1M input, $60/1M output (example, check latest pricing)

        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            model: Model used (defaults to self.model)

        Returns:
            Estimated cost in USD
        """
        effective_model = model or self.model

        # Pricing table (update as needed)
        pricing = {
            "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
            "gpt-4": {"input": 30.0, "output": 60.0},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50}
        }

        # Default pricing for unknown models (use gpt-4.1-mini rates)
        rates = pricing.get(effective_model, pricing["gpt-4.1-mini"])

        input_cost = (prompt_tokens / 1_000_000) * rates["input"]
        output_cost = (completion_tokens / 1_000_000) * rates["output"]

        return round(input_cost + output_cost, 6)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get client usage statistics.

        Returns:
            Dictionary with:
                - total_calls: Total number of API calls made
                - total_prompt_tokens: Total input tokens used
                - total_completion_tokens: Total output tokens used
                - total_errors: Total number of errors encountered
                - estimated_cost: Estimated total cost in USD
        """
        return {
            "total_calls": self._total_calls,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_errors": self._total_errors,
            "estimated_cost": self.estimate_cost(
                self._total_prompt_tokens,
                self._total_completion_tokens
            )
        }

    def reset_statistics(self) -> None:
        """Reset usage statistics to zero."""
        self._total_calls = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_errors = 0


def get_default_client() -> LLMClient:
    """
    Get a singleton instance of the default LLM client.

    This is useful for sharing a single client across multiple modules
    to consolidate statistics and logging.

    Returns:
        Shared LLMClient instance
    """
    global _default_client
    if '_default_client' not in globals():
        _default_client = LLMClient()
    return _default_client
