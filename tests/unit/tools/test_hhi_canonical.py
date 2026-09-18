"""
HHI 정본 구현 테스트 (§1.1 / D1)

- 정본 스케일은 0-1. 0-10000 포인트가 필요한 소비처는 hhi_to_points()를 쓴다.
- 자체 HHI 계산을 다시 만드는 회귀를 소스 스캔으로 막는다.
"""

import re
from pathlib import Path

import pytest

from src.tools.calculators.metric_calculator import (
    MetricCalculator,
    calculate_hhi_from_counts,
    count_brands,
    hhi_to_points,
)

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"


class TestCanonicalScale:
    """정본 스케일 0-1 고정"""

    @pytest.mark.parametrize(
        "counts",
        [
            {"A": 100},
            {"A": 50, "B": 50},
            {"A": 1, "B": 1, "C": 1, "D": 1},
            {f"B{i}": 1 for i in range(100)},
            {"A": 97, "B": 2, "C": 1},
        ],
    )
    def test_always_between_zero_and_one(self, counts):
        hhi = calculate_hhi_from_counts(counts)
        assert 0.0 <= hhi <= 1.0

    def test_monopoly_is_one(self):
        assert calculate_hhi_from_counts({"A": 42}) == 1.0

    def test_empty_is_zero(self):
        assert calculate_hhi_from_counts({}) == 0.0
        assert calculate_hhi_from_counts({"A": 0}) == 0.0

    def test_equal_split(self):
        assert calculate_hhi_from_counts({"A": 25, "B": 25, "C": 25, "D": 25}) == 0.25

    def test_metric_calculator_uses_canonical(self):
        calc = MetricCalculator(config={})
        records = [{"brand": "A", "rank": 1}, {"brand": "B", "rank": 2}]
        assert calc.calculate_hhi(records) == calculate_hhi_from_counts({"A": 1, "B": 1})


class TestPointsConversion:
    """0-10000 포인트 변환 헬퍼"""

    def test_round_numbers(self):
        assert hhi_to_points(0.0) == 0
        assert hhi_to_points(1.0) == 10000
        assert hhi_to_points(0.25) == 2500

    def test_is_int(self):
        assert isinstance(hhi_to_points(0.1234), int)

    def test_roundtrip_within_tolerance(self):
        for hhi in (0.0, 0.05, 0.1234, 0.5, 1.0):
            assert abs(hhi_to_points(hhi) / 10000 - hhi) < 1e-4


class TestCountBrands:
    def test_excludes_unknown_variants(self):
        records = [
            {"brand": "LANEIGE"},
            {"brand": "Unknown"},
            {"brand": "unknown"},
            {"brand": "N/A"},
            {"brand": ""},
            {"brand": None},
            {"brand": "  "},
        ]
        assert count_brands(records) == {"LANEIGE": 1}

    def test_can_include_unknown(self):
        records = [{"brand": "A"}, {"brand": "Unknown"}]
        assert count_brands(records, exclude_unknown=False) == {"A": 1, "Unknown": 1}

    def test_strips_whitespace(self):
        assert count_brands([{"brand": " LANEIGE "}, {"brand": "LANEIGE"}]) == {"LANEIGE": 2}

    def test_custom_brand_key(self):
        assert count_brands([{"b": "A"}], brand_key="b") == {"A": 1}


class TestNoDuplicateImplementations:
    """두 스케일 혼용 회귀 방지 — src/에 자체 HHI 계산이 재등장하지 않는다"""

    # 정본 구현이 사는 파일만 예외
    ALLOWED = {"src/tools/calculators/metric_calculator.py"}

    # `(count / total * 100) ** 2` / `share ** 2 * 10000` 류의 직접 계산
    PATTERNS = [
        re.compile(r"\*\s*100\s*\)\s*\*\*\s*2"),
        re.compile(r"\*\*\s*2\s*for\s+\w+\s+in\s+\w*(?:count|share|brand)\w*"),
        re.compile(r"sum\(\s*s\s*\*\s*s\s+for"),
    ]

    def test_no_inline_hhi_math(self):
        offenders = []
        for path in SRC_ROOT.rglob("*.py"):
            rel = path.relative_to(SRC_ROOT.parent).as_posix()
            if rel in self.ALLOWED:
                continue
            text = path.read_text(encoding="utf-8")
            if "hhi" not in text.lower():
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if any(p.search(line) for p in self.PATTERNS):
                    offenders.append(f"{rel}:{lineno}: {line.strip()}")
        assert not offenders, (
            "HHI 자체 계산이 재등장했습니다. "
            "metric_calculator.calculate_hhi_from_counts()를 쓰세요:\n" + "\n".join(offenders)
        )
