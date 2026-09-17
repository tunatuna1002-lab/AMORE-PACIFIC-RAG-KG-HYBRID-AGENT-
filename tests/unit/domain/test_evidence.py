"""증거 카드(Evidence) 도메인 모델 — 설계 E1·E8 (docs/plans/evidence-react-ontology-kickoff-prompt-2026-09-17.md)

KG 사실·DB 수치·문서 청크·규칙 추론·도구 관찰을 하나의 카드로 표현한다. 답변이 카드 id를
인용하므로 id는 짧고, 같은 사실이면 실행·프로세스가 달라도 같아야 한다(sha1, hash() 금지).
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.domain.entities.evidence import (
    ID_BASIS_KEY,
    ID_HEX_LENGTH,
    KNOWN_UNITS,
    Evidence,
    EvidenceKind,
    EvidenceSet,
    EvidenceUnit,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _sos(subject: str = "laneige", value: float = 0.02, as_of: str = "2026-08-31") -> Evidence:
    return Evidence.create(
        kind=EvidenceKind.METRIC,
        subject=subject,
        predicate="sos",
        object="lip_care",
        value=value,
        unit=EvidenceUnit.RATIO,
        as_of=as_of,
        source="sqlite:brand_metrics",
        confidence=1.0,
        text="LANEIGE lip_care SoS 2%",
        metadata={"display_name": "LANEIGE"},
    )


class TestId:
    def test_same_fact_same_id(self):
        assert _sos().id == _sos().id

    def test_id_format_is_prefix_and_six_hex(self):
        card = _sos()
        prefix, hex_part = card.id.split("-")
        assert prefix == "M"
        assert len(hex_part) == ID_HEX_LENGTH == 6
        int(hex_part, 16)

    @pytest.mark.parametrize(
        ("kind", "prefix"),
        [
            (EvidenceKind.RELATION, "R"),
            (EvidenceKind.METRIC, "M"),
            (EvidenceKind.DOCUMENT, "D"),
            (EvidenceKind.INFERENCE, "I"),
            (EvidenceKind.OBSERVATION, "O"),
        ],
    )
    def test_prefix_per_kind(self, kind, prefix):
        card = Evidence.create(kind=kind, subject="s", predicate="p", source="x", text="t")
        assert card.id.startswith(f"{prefix}-")

    def test_id_is_sha1_of_identity_fields(self):
        """프로세스 무관 결정성: 정규화 문자열의 sha1 앞 6자리여야 한다."""
        card = _sos()
        identity = json.dumps(
            [
                "metric",
                "laneige",
                "sos",
                "lip_care",
                0.02,
                "ratio",
                "2026-08-31",
                "sqlite:brand_metrics",
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        assert card.identity_key() == identity
        assert card.id == "M-" + hashlib.sha1(identity.encode("utf-8")).hexdigest()[:6]

    def test_id_stable_across_processes(self):
        """PYTHONHASHSEED가 달라도 같은 id — 내장 hash()를 쓰면 여기서 깨진다."""
        code = (
            "from src.domain.entities.evidence import Evidence, EvidenceKind;"
            "print(Evidence.create(kind=EvidenceKind.RELATION, subject='laneige',"
            " predicate='competesWith', object='cosrx', source='kg', text='t').id)"
        )
        ids = set()
        for seed in ("1", "2"):
            out = subprocess.run(
                [sys.executable, "-c", code],
                cwd=REPO_ROOT,
                env={"PYTHONHASHSEED": seed, "PATH": ""},
                capture_output=True,
                text=True,
                check=True,
            )
            ids.add(out.stdout.strip())
        local = Evidence.create(
            kind=EvidenceKind.RELATION,
            subject="laneige",
            predicate="competesWith",
            object="cosrx",
            source="kg",
            text="t",
        ).id
        assert ids == {local}

    @pytest.mark.parametrize(
        "change",
        [
            {"subject": "cosrx"},
            {"value": 0.03},
            {"as_of": "2026-09-01"},
        ],
    )
    def test_different_fact_different_id(self, change):
        base = _sos()
        other = _sos(**change)
        assert base.id != other.id

    def test_display_fields_do_not_change_id(self):
        """text·detail·metadata·confidence는 사실의 정체가 아니다."""
        a = _sos()
        b = Evidence.create(
            kind=EvidenceKind.METRIC,
            subject="laneige",
            predicate="sos",
            object="lip_care",
            value=0.02,
            unit=EvidenceUnit.RATIO,
            as_of="2026-08-31",
            source="sqlite:brand_metrics",
            confidence=0.5,
            text="다른 문장",
            detail="본문",
            metadata={"display_name": "laneige"},
        )
        assert a.id == b.id

    def test_integral_float_and_int_share_id(self):
        a = Evidence.create(
            kind=EvidenceKind.METRIC, subject="s", predicate="rank", value=2, source="x", text="t"
        )
        b = Evidence.create(
            kind=EvidenceKind.METRIC, subject="s", predicate="rank", value=2.0, source="x", text="t"
        )
        assert a.id == b.id

    def test_bool_and_int_do_not_collide(self):
        a = Evidence.create(
            kind=EvidenceKind.METRIC, subject="s", predicate="p", value=True, source="x", text="t"
        )
        b = Evidence.create(
            kind=EvidenceKind.METRIC, subject="s", predicate="p", value=1, source="x", text="t"
        )
        assert a.id != b.id

    def test_id_basis_overrides_fields(self):
        """문서 카드는 chunk_id 기반 — 본문·점수가 달라도 같은 청크면 같은 id."""
        a = Evidence.create(
            kind=EvidenceKind.DOCUMENT,
            subject="metric_guide",
            predicate="states",
            source="chroma",
            text="t1",
            id_basis="metric_guide_1",
        )
        b = Evidence.create(
            kind=EvidenceKind.DOCUMENT,
            subject="other",
            predicate="states",
            source="bm25",
            text="t2",
            id_basis="metric_guide_1",
        )
        assert a.id == b.id
        assert a.metadata[ID_BASIS_KEY] == "metric_guide_1"


class TestValidation:
    def test_frozen(self):
        card = _sos()
        with pytest.raises(ValidationError):
            card.value = 0.5

    def test_rejects_forged_id(self):
        data = _sos().model_dump()
        data["id"] = "M-000000"
        with pytest.raises(ValidationError):
            Evidence.model_validate(data)

    def test_rejects_prefix_kind_mismatch(self):
        data = _sos().model_dump()
        data["id"] = "R-" + data["id"][2:]
        with pytest.raises(ValidationError):
            Evidence.model_validate(data)

    @pytest.mark.parametrize("as_of", ["2026/08/31", "2026-13-01", "20260831", "2026-08-31T00:00"])
    def test_rejects_bad_as_of(self, as_of):
        with pytest.raises(ValidationError):
            _sos(as_of=as_of)

    @pytest.mark.parametrize("confidence", [-0.1, 1.1])
    def test_rejects_confidence_out_of_range(self, confidence):
        with pytest.raises(ValidationError):
            Evidence.create(
                kind=EvidenceKind.INFERENCE,
                subject="s",
                predicate="p",
                source="rule:x",
                text="t",
                confidence=confidence,
            )

    def test_rejects_unknown_unit(self):
        with pytest.raises(ValidationError):
            Evidence.create(
                kind=EvidenceKind.METRIC,
                subject="s",
                predicate="sos",
                value=2.0,
                unit="percent",
                source="x",
                text="t",
            )

    def test_rejects_nan(self):
        with pytest.raises(ValidationError):
            _sos(value=float("nan"))

    def test_known_units_are_constants(self):
        assert EvidenceUnit.RATIO in KNOWN_UNITS
        assert EvidenceUnit.INDEX_0_1 in KNOWN_UNITS

    def test_value_types_preserved(self):
        for value in (True, 3, 0.5, "dominant"):
            card = Evidence.create(
                kind=EvidenceKind.METRIC,
                subject="s",
                predicate="p",
                value=value,
                source="x",
                text="t",
            )
            assert card.value == value
            assert type(card.value) is type(value)


class TestSerialization:
    def test_json_round_trip(self):
        card = Evidence.create(
            kind=EvidenceKind.INFERENCE,
            subject="laneige",
            predicate="market_dominance",
            object="lip_care",
            value="dominant_in_fragmented",
            as_of="2026-08-31",
            source="rule:market_dominance_fragmented",
            confidence=0.9,
            derived_from=["M-aaaaaa", "M-bbbbbb"],
            text="LANEIGE는 분산 시장에서 강한 존재감",
            detail="권장: 포지션 유지",
            metadata={"satisfied_conditions": ["sos_above_0.15"]},
        )
        restored = Evidence.model_validate_json(card.model_dump_json())
        assert restored == card
        assert restored.derived_from == ("M-aaaaaa", "M-bbbbbb")

    def test_dict_round_trip_bool_value(self):
        card = Evidence.create(
            kind=EvidenceKind.METRIC,
            subject="laneige",
            predicate="present_in_top100",
            object="lip_care",
            value=False,
            unit=EvidenceUnit.BOOLEAN,
            as_of="2026-08-31",
            source="sqlite:brand_metrics",
            text="t",
        )
        restored = Evidence.model_validate(json.loads(card.model_dump_json()))
        assert restored.value is False
        assert restored.id == card.id


class TestEvidenceSet:
    def test_preserves_order_and_dedupes(self):
        a, b = _sos("laneige"), _sos("cosrx")
        dup = _sos("laneige")
        cards = EvidenceSet([a, b, dup])
        assert [c.id for c in cards] == [a.id, b.id]
        assert len(cards) == 2
        assert a.id in cards

    def test_first_wins_on_duplicate(self):
        first = _sos()
        second = Evidence.create(**{**_sos().model_dump(exclude={"id"}), "text": "두 번째"})
        cards = EvidenceSet([first])
        assert cards.add(second) is first
        assert cards.by_id(first.id).text == first.text

    def test_by_id_and_of_kind(self):
        metric = _sos()
        relation = Evidence.create(
            kind=EvidenceKind.RELATION,
            subject="laneige",
            predicate="ownedBy",
            object="amorepacific",
            source="kg",
            text="t",
        )
        cards = EvidenceSet([relation, metric])
        assert cards.by_id(metric.id) is metric
        assert cards.by_id("M-ffffff") is None
        assert cards.of_kind(EvidenceKind.METRIC) == [metric]
        assert cards.of_kind(EvidenceKind.METRIC, EvidenceKind.RELATION) == [relation, metric]

    def test_collision_extends_id(self):
        """같은 6자리 id에 다른 사실이 오면 뒤에 온 카드의 id를 7자리 이상으로 늘린다."""
        seen: dict[str, Evidence] = {}
        pair = None
        for i in range(200_000):
            card = Evidence.create(
                kind=EvidenceKind.RELATION, subject=f"s{i}", predicate="p", source="x", text="t"
            )
            if card.id in seen:
                pair = (seen[card.id], card)
                break
            seen[card.id] = card
        assert pair is not None, "6자리 충돌 쌍을 찾지 못함"
        first, second = pair

        cards = EvidenceSet([first])
        stored = cards.add(second)
        assert stored.id != first.id
        assert stored.id.startswith(first.id)
        assert len(stored.id) == len(first.id) + 1
        assert stored.identity_key() == second.identity_key()
        # 확장된 id도 검증을 통과하는 정식 카드다
        assert Evidence.model_validate(stored.model_dump()) == stored
        # 같은 사실을 다시 넣으면 확장된 카드를 돌려준다
        assert cards.add(second) is stored
        assert len(cards) == 2
        assert cards.by_id(stored.id) is stored
