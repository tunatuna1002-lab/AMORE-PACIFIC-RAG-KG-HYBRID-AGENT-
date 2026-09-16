"""
Design system constants
=======================
AMOREPACIFIC CI 색상·폰트·로고 경로. 리포트 세 포맷이 공유한다.
"""

from __future__ import annotations

from docx.shared import Cm, Pt, RGBColor

# =============================================================================
# Design System Constants
# =============================================================================


class DesignSystem:
    """AMOREPACIFIC 디자인 시스템"""

    # Colors
    PACIFIC_BLUE = RGBColor(0, 28, 88)  # #001C58
    AMORE_BLUE = RGBColor(31, 87, 149)  # #1F5795
    GRAY = RGBColor(125, 125, 125)  # #7D7D7D
    LIGHT_GRAY = RGBColor(245, 245, 245)  # #F5F5F5
    WHITE = RGBColor(255, 255, 255)  # #FFFFFF
    ACCENT_RED = RGBColor(229, 57, 53)  # #E53935
    ACCENT_GREEN = RGBColor(67, 160, 71)  # #43A047

    # Font Family - Arita Dotum (돋움, 산세리프 - 제목/목차용)
    FONT_DOTUM = "Arita Dotum KR"
    FONT_DOTUM_MEDIUM = "Arita Dotum KR Medium"
    FONT_DOTUM_SEMIBOLD = "Arita Dotum KR SemiBold"
    FONT_DOTUM_BOLD = "Arita Dotum KR Bold"
    FONT_DOTUM_LIGHT = "Arita Dotum KR Light"

    # Font Family - Arita Buri (부리, 세리프 - 본문용)
    FONT_BURI = "Arita Buri KR"
    FONT_BURI_MEDIUM = "Arita Buri KR Medium"
    FONT_BURI_SEMIBOLD = "Arita Buri KR SemiBold"
    FONT_BURI_BOLD = "Arita Buri KR Bold"
    FONT_BURI_LIGHT = "Arita Buri KR Light"

    # 레거시 호환성
    FONT_FAMILY = FONT_DOTUM
    FONT_FAMILY_MEDIUM = FONT_DOTUM_MEDIUM
    FONT_FAMILY_SEMIBOLD = FONT_DOTUM_SEMIBOLD
    FONT_FAMILY_BOLD = FONT_DOTUM_BOLD
    FONT_FAMILY_LIGHT = FONT_DOTUM_LIGHT
    FONT_FAMILY_FALLBACK = "Malgun Gothic"  # Windows fallback
    FONT_FAMILY_MAC = "Apple SD Gothic Neo"  # macOS fallback

    # Typography (points)
    TITLE_SIZE = Pt(28)
    SUBTITLE_SIZE = Pt(16)
    HEADING1_SIZE = Pt(18)
    HEADING2_SIZE = Pt(14)
    HEADING3_SIZE = Pt(12)
    BODY_SIZE = Pt(11)
    CAPTION_SIZE = Pt(9)

    # Spacing
    PAGE_MARGIN_TOP = Cm(2)
    PAGE_MARGIN_BOTTOM = Cm(2)
    PAGE_MARGIN_LEFT = Cm(2.5)
    PAGE_MARGIN_RIGHT = Cm(2.5)

    # Logo paths (relative to project root)
    LOGO_COLOR_EN = "Amorepacific_CI_wordmark_v1.00/png/Color/Amorepacific_Wordmark_Color_En.png"
    LOGO_BASIC_EN = "Amorepacific_CI_wordmark_v1.00/png/Basic/Amorepacific_Wordmark_Basic_En.png"
    LOGO_REVERSE_EN = (
        "Amorepacific_CI_wordmark_v1.00/png/Reverse/Amorepacific_Wordmark_Reverse_En.png"
    )
