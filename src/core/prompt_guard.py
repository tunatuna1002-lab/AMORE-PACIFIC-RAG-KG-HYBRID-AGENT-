"""
프롬프트 인젝션 방어 시스템 (PromptGuard)
=========================================
3중 보안 레이어:
- Layer 1: 입력 필터링 (인젝션 패턴 차단)
- Layer 2: 범위 제한 (Out-of-scope 감지)
- Layer 3: 출력 검증 (민감 정보 마스킹)

연결 파일:
- core/brain.py: UnifiedBrain에서 사용
"""

import logging
import re

logger = logging.getLogger(__name__)


class PromptGuard:
    """프롬프트 인젝션 방어 시스템"""

    # Layer 1: 입력 필터링 패턴 (명백한 공격 차단)
    INJECTION_PATTERNS = [
        # === 영문 직접 지시 무시 시도 ===
        r"(?i)ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)",
        r"(?i)disregard\s+(all\s+)?(previous|prior|above|earlier)",
        r"(?i)forget\s+(all\s+)?(previous|prior|above|everything)",
        r"(?i)override\s+(all\s+)?(previous|prior|system)",
        # === 한국어 직접 지시 무시 시도 ===
        r"(이전|모든)\s*(지시|규칙|지시사항)을?\s*무시",
        r"(원래|기존)\s*(지시사항|지시|설정)을?\s*(알려|보여|출력)",
        r"내가\s*시킨\s*대로\s*해",
        # === 영문 시스템 프롬프트 탈취 시도 ===
        r"(?i)(show|tell|reveal|display|print|output)\s+(me\s+)?(the\s+)?(your\s+)?(system\s+)?prompt",
        r"(?i)(show|tell|reveal|display|print|output)\s+(me\s+)?(your\s+)?(initial\s+)?(instructions?|rules?|configuration)",
        r"(?i)what\s+(are|is)\s+(your\s+)?(system\s+)?(prompt|instructions?)",
        r"(?i)(repeat|echo)\s+(the\s+)?(your\s+)?(system\s+)?(prompt|instructions?)",
        # === 한국어 시스템 프롬프트 탈취 시도 ===
        r"시스템\s*프롬프트",
        r"(?i)system\s+prompt를?\s+.{0,10}(출력|보여|알려)",
        r"(?i)system\s+prompt.{0,20}(출력|보여|알려|보내)",
        r"너의\s*(설정|지시|명령)을?\s*(알려|보여|출력)",
        # === 영문 역할 탈취 시도 ===
        r"(?i)you\s+are\s+now\s+(a|an|in)\s+",
        r"(?i)act\s+as\s+(a|an)\s+(?!laneige|amazon|market)",
        r"(?i)pretend\s+(to\s+be|you('re|are))",
        r"(?i)roleplay\s+as",
        r"(?i)switch\s+(to\s+)?(a\s+)?(different\s+)?(role|mode|persona|unrestricted)",
        r"(?i)no\s+(content\s+)?policy",
        # === 한국어 역할 탈취 시도 ===
        r"새로운\s*역할",
        r"제한\s*없는\s*(AI|어시스턴트|모드)",
        # === 컨텍스트 혼동 시도 ===
        r"(?i)---\s*(end|start)\s+(of\s+)?(new\s+)?(system|prompt|instructions?)",
        r"(?i)\[\s*(system|end|new)\s*(prompt|instructions?)?\s*\]",
        r"(?i)<\s*/?\s*(system|prompt|instructions?)\s*>",
        # === 인코딩 우회 시도 ===
        r"(?i)(decode|translate|interpret)\s+(this\s+)?(base64|hex|rot13|binary)",
        r"(?i)base64[:\s]",
        # === DAN/탈옥 시도 ===
        r"(?i)\bDAN\b",
        r"(?i)jailbreak",
        r"(?i)developer\s+mode",
        r"(?i)unrestricted\s+mode",
    ]

    # 범위 외 주제 키워드 (경고 수준)
    OUT_OF_SCOPE_KEYWORDS = [
        # 일반 주제
        "날씨",
        "weather",
        "기온",
        "비",
        "눈",
        "정치",
        "politics",
        "대통령",
        "선거",
        "국회",
        "스포츠",
        "축구",
        "야구",
        "basketball",
        "football",
        "주식",
        "stock",
        "비트코인",
        "bitcoin",
        "crypto",
        "투자",
        "영화",
        "movie",
        "드라마",
        "netflix",
        "게임",
        "game",
        "맛집",
        "restaurant",
        "요리",
        "recipe",
        "음식",
        "연예인",
        "celebrity",
        "아이돌",
        "idol",
        # 유해 콘텐츠
        "폭탄",
        "bomb",
        "해킹",
        "hacking",
        "exploit",
        "마약",
        "drug",
        "불법",
        "illegal",
    ]

    # 민감 정보 패턴 (출력 검증용)
    SENSITIVE_OUTPUT_PATTERNS = [
        r"(?i)system\s*prompt",
        r"(?i)당신은.*전문가입니다",  # 시스템 프롬프트 시작 부분
        r"(?i)namespace\s+functions",  # 도구 정의 노출
        r"(?i)type\s+\w+\s*=\s*\(\s*_\s*:",  # TypeScript 함수 정의
        r"(?i)api[_\s]?key",
        r"(?i)password",
        r"(?i)secret",
        r"(?i)credential",
        # === 클라우드/서비스 자격증명 ===
        r"AKIA[0-9A-Z]{16}",  # AWS Access Key ID
        r"(?i)aws.{0,20}['\"][0-9a-zA-Z/+]{40}['\"]",  # AWS Secret Key
        r"gh[pousr]_[A-Za-z0-9_]{36,255}",  # GitHub PAT
        r"xox[bporas]-[0-9a-zA-Z-]{10,}",  # Slack Token
        r"-----BEGIN\s+(RSA\s+|EC\s+|DSA\s+|OPENSSH\s+)?PRIVATE\s+KEY-----",
    ]

    @classmethod
    def check_input(cls, text: str) -> tuple[bool, str | None, str]:
        """
        입력 텍스트 검증 (Layer 1)

        Returns:
            (is_safe, block_reason, sanitized_text)
        """
        # 0. NFKC 유니코드 정규화 (유니코드 우회 공격 방어)
        import unicodedata

        text = unicodedata.normalize("NFKC", text)

        # 1. 명백한 인젝션 패턴 검사
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, text):
                logger.warning(f"Injection attempt blocked: pattern matched - {pattern[:50]}")
                return False, "injection_detected", ""

        # 2. 범위 외 키워드 검사 (차단하지 않고 플래그만)
        # 비즈니스 관련 키워드가 포함되면 범위 외 판단 건너뛰기
        business_keywords = [
            "laneige",
            "cosrx",
            "amazon",
            "sos",
            "hhi",
            "cpi",
            "lip care",
            "lip sleeping",
            "순위",
            "점유율",
            "경쟁",
            "분석",
            "카테고리",
            "브랜드",
            "market",
            "rank",
            "share",
        ]
        text_lower = text.lower()
        has_business_context = any(bk in text_lower for bk in business_keywords)

        if not has_business_context:
            for keyword in cls.OUT_OF_SCOPE_KEYWORDS:
                kw_lower = keyword.lower()
                # 한글 키워드: 2글자 이상만 부분 매칭, 1글자는 정확 매칭
                if len(kw_lower) <= 1:
                    # 1글자 한글 (비, 눈 등)은 단독 단어일 때만 매칭
                    if re.search(rf"(?:^|[\s,.]){re.escape(kw_lower)}(?:$|[\s,.])", text_lower):
                        return True, "out_of_scope_warning", text
                elif kw_lower in text_lower:
                    return True, "out_of_scope_warning", text

        return True, None, text

    @classmethod
    def check_output(cls, text: str) -> tuple[bool, str]:
        """
        출력 텍스트 검증 (Layer 3)

        Returns:
            (is_safe, sanitized_text)
        """
        # 시스템 프롬프트 전체 노출 감지 (먼저 체크 — regex 치환 전)
        text_lower = text.lower()
        if "namespace functions" in text_lower or "type get_brand_status" in text_lower:
            logger.warning("System prompt leak detected - blocking response")
            return (
                False,
                "시스템 정보는 공개할 수 없습니다. LANEIGE 마켓 분석에 관해 질문해 주세요.",
            )

        # 민감 정보 노출 검사 (키워드 마스킹)
        for pattern in cls.SENSITIVE_OUTPUT_PATTERNS:
            if re.search(pattern, text):
                text = re.sub(pattern, "[REDACTED]", text)
                logger.warning(f"Sensitive output detected and redacted: {pattern[:30]}")

        return True, text

    @classmethod
    def get_rejection_message(cls, reason: str) -> str:
        """차단 사유별 응답 메시지"""
        messages = {
            "injection_detected": (
                "죄송합니다. 해당 요청은 처리할 수 없습니다.\n\n"
                "저는 LANEIGE 브랜드의 Amazon US 마켓 분석을 돕는 전문 어시스턴트입니다.\n"
                "브랜드 순위, 경쟁사 분석, 제품 성과 등에 대해 질문해 주세요."
            ),
            "out_of_scope": (
                "해당 주제는 제 전문 영역이 아닙니다.\n\n"
                "저는 LANEIGE 브랜드의 Amazon US 마켓 분석 전문가입니다.\n"
                "다음과 같은 질문에 답변드릴 수 있습니다:\n"
                "• LANEIGE 현재 순위 및 성과\n"
                "• 경쟁사 대비 분석\n"
                "• 카테고리별 트렌드\n"
                "• 제품별 상세 분석"
            ),
        }
        return messages.get(reason, messages["out_of_scope"])
