"""
입력 검증기 - 프롬프트 인젝션 방어 및 입력 살균

이 모듈은 사용자 입력을 검증하고 살균하여 보안을 강화합니다.

사용 예:
    from src.api.validators.input_validator import InputValidator
    from src.domain.exceptions import DataValidationError

    validator = InputValidator()
    try:
        is_valid, sanitized = validator.validate(user_input)
        # sanitized 사용
    except DataValidationError as e:
        # 인젝션 탐지 또는 길이 초과
        return error_response(e)
"""
import re
from typing import Tuple

from src.domain.exceptions import DataValidationError


class InputValidator:
    """
    사용자 입력 검증기

    - 프롬프트 인젝션 탐지
    - 입력 길이 제한
    - HTML/스크립트 살균
    """

    MAX_LENGTH = 2000

    # 프롬프트 인젝션 패턴 (정규식)
    # 대소문자 무관하게 매칭됨
    INJECTION_PATTERNS = [
        r'ignore\s+(all\s+)?previous\s+instructions?',
        r'disregard\s+(all\s+)?(above|previous)',
        r'^system\s*:',  # 문자열 시작에서만
        r'you\s+are\s+now\s+',
        r'pretend\s+to\s+be',
        r'act\s+as\s+(if|a|an)\s+',
        r'forget\s+everything',
        r'new\s+instructions?\s*:',
        r'jailbreak',
        r'bypass\s+(all\s+)?restrictions?',
        r'override\s+(your\s+)?(instructions?|rules?)',
    ]

    def __init__(self):
        """검증기 초기화"""
        # 패턴을 미리 컴파일하여 성능 향상
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.INJECTION_PATTERNS
        ]

    def validate(self, text: str) -> Tuple[bool, str]:
        """
        입력 텍스트 검증 및 살균

        Args:
            text: 검증할 텍스트

        Returns:
            (is_valid, sanitized_text) 튜플
            - is_valid: 검증 통과 여부
            - sanitized_text: 살균된 텍스트

        Raises:
            DataValidationError: 인젝션 탐지 또는 길이 초과 시
        """
        # 1. 길이 제한 검사
        if len(text) > self.MAX_LENGTH:
            raise DataValidationError(
                f"Input exceeds {self.MAX_LENGTH} characters",
                field="message",
                value=len(text),
                constraint=f"max_length={self.MAX_LENGTH}"
            )

        # 2. 프롬프트 인젝션 탐지
        text_for_check = text.lower()
        for pattern in self._compiled_patterns:
            if pattern.search(text_for_check):
                raise DataValidationError(
                    "Potential prompt injection detected",
                    field="message",
                    value=text[:100] if len(text) > 100 else text,
                    constraint="no_injection_patterns"
                )

        # 3. HTML 태그 제거 (XSS 방지)
        sanitized = re.sub(r'<[^>]+>', '', text)

        # 4. 스크립트 관련 문자열 제거
        sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'on\w+\s*=', '', sanitized, flags=re.IGNORECASE)

        # 5. 앞뒤 공백 제거
        sanitized = sanitized.strip()

        return True, sanitized

    def is_safe(self, text: str) -> bool:
        """
        입력이 안전한지 확인 (예외 발생 없이)

        Args:
            text: 확인할 텍스트

        Returns:
            안전하면 True, 위험하면 False
        """
        try:
            self.validate(text)
            return True
        except DataValidationError:
            return False

    def sanitize_only(self, text: str) -> str:
        """
        검증 없이 살균만 수행

        Args:
            text: 살균할 텍스트

        Returns:
            살균된 텍스트
        """
        # HTML 태그 제거
        sanitized = re.sub(r'<[^>]+>', '', text)

        # 스크립트 관련 문자열 제거
        sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'on\w+\s*=', '', sanitized, flags=re.IGNORECASE)

        return sanitized.strip()


# 편의를 위한 싱글톤 인스턴스
_validator_instance = None


def get_validator() -> InputValidator:
    """
    InputValidator 싱글톤 인스턴스 반환

    Returns:
        InputValidator 인스턴스
    """
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = InputValidator()
    return _validator_instance


def validate_input(text: str) -> Tuple[bool, str]:
    """
    입력 검증 편의 함수

    Args:
        text: 검증할 텍스트

    Returns:
        (is_valid, sanitized_text) 튜플

    Raises:
        DataValidationError: 검증 실패 시
    """
    return get_validator().validate(text)
