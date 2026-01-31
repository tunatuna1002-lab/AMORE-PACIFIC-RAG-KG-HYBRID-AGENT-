"""
입력 검증기 - Prompt Injection 방어

Audit Report C.2 대응: 시스템 프롬프트/API 키 노출 방지
"""

import re

from src.domain.exceptions import DataValidationError


class InputValidator:
    """사용자 입력 검증 및 살균"""

    MAX_LENGTH = 2000

    # Prompt Injection 패턴 (대소문자 무시)
    INJECTION_PATTERNS = [
        # Ignore/Disregard 패턴
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"disregard\s+(all\s+)?above",
        r"disregard\s+(all\s+)?previous",
        # System override 패턴
        r"^system\s*:",
        r"system\s*:\s*override",
        r"system\s*:\s*reveal",
        # Jailbreak 패턴
        r"you\s+are\s+now\s+",
        r"pretend\s+to\s+be",
        r"act\s+as\s+if",
        r"forget\s+everything",
        r"new\s+instructions\s*:",
        # 시스템 프롬프트 노출 시도
        r"reveal\s+your\s+(system\s+)?prompt",
        r"show\s+me\s+your\s+instructions",
        r"what\s+are\s+your\s+rules",
    ]

    # HTML 태그 패턴
    HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

    def validate(self, text: str) -> tuple[bool, str]:
        """
        입력 텍스트 검증 및 살균

        Args:
            text: 검증할 입력 텍스트

        Returns:
            Tuple[bool, str]: (검증 통과 여부, 살균된 텍스트)

        Raises:
            DataValidationError: 검증 실패 시
        """
        if not text:
            raise DataValidationError(
                "Empty input", field="message", value="", constraint="non_empty"
            )

        # 1. 길이 제한
        if len(text) > self.MAX_LENGTH:
            raise DataValidationError(
                f"Input exceeds {self.MAX_LENGTH} characters",
                field="message",
                value=len(text),
                constraint=f"max_length={self.MAX_LENGTH}",
            )

        # 2. Prompt Injection 탐지
        text_lower = text.lower()
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                raise DataValidationError(
                    "Potential prompt injection detected",
                    field="message",
                    value=text[:100] + "..." if len(text) > 100 else text,
                    constraint="no_injection_patterns",
                )

        # 3. HTML 태그 제거 (XSS 방지)
        sanitized = self.HTML_TAG_PATTERN.sub("", text)

        # 4. 앞뒤 공백 제거
        sanitized = sanitized.strip()

        return True, sanitized

    def is_safe(self, text: str) -> bool:
        """
        입력이 안전한지 확인 (예외 발생 없이)

        Args:
            text: 검증할 입력 텍스트

        Returns:
            bool: 안전 여부
        """
        try:
            self.validate(text)
            return True
        except DataValidationError:
            return False
