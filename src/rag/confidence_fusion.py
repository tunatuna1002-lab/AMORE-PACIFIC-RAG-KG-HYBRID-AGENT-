"""
Confidence Fusion Module
다중 소스 (벡터 검색, 온톨로지 추론, 엔티티 연결) 신뢰도 통합

Fusion Strategy:
- 가중치 기반 융합 (Vector: 0.40, Ontology: 0.35, Entity: 0.25)
- 점수 정규화 (Min-Max or Softmax)
- 설명 가능한 결과 생성
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import numpy as np


class ScoreNormalizationMethod(str, Enum):
    """점수 정규화 방법"""
    MIN_MAX = "min_max"          # Min-Max 정규화
    SOFTMAX = "softmax"          # Softmax (확률분포)
    Z_SCORE = "z_score"          # Z-Score 표준화
    NONE = "none"                # 정규화 없음


class FusionStrategy(str, Enum):
    """융합 전략"""
    WEIGHTED_SUM = "weighted_sum"        # 가중합
    HARMONIC_MEAN = "harmonic_mean"      # 조화평균 (모든 점수가 높아야 높음)
    GEOMETRIC_MEAN = "geometric_mean"    # 기하평균
    MAX_SCORE = "max_score"              # 최대값
    MIN_SCORE = "min_score"              # 최소값 (보수적)


@dataclass
class SearchResult:
    """벡터/키워드 검색 결과"""
    content: str
    score: float                    # 0~1 (유사도/관련성)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "vector"          # vector, keyword, hybrid


@dataclass
class InferenceResult:
    """온톨로지 추론 결과"""
    insight: str
    confidence: float               # 0~1 (추론 신뢰도)
    evidence: Dict[str, Any] = field(default_factory=dict)
    rule_name: Optional[str] = None


@dataclass
class LinkedEntity:
    """엔티티 연결 결과"""
    entity_id: str
    entity_name: str
    entity_type: str                # Brand, Product, Category
    link_confidence: float          # 0~1 (연결 강도)
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceScore:
    """각 소스별 점수"""
    source_name: str                # vector, ontology, entity
    raw_score: float                # 원본 점수
    normalized_score: float         # 정규화 점수
    weight: float                   # 가중치
    contribution: float             # 최종 기여도 (normalized * weight)
    confidence_level: str           # high, medium, low
    explanation: str                # 해당 소스 점수 설명


@dataclass
class FusedResult:
    """통합된 최종 결과"""
    documents: List[Dict[str, Any]]  # 통합 문서들
    confidence: float                # 최종 신뢰도 (0~1)
    explanation: str                 # 종합 설명
    source_scores: List[SourceScore] # 소스별 점수 상세
    fusion_strategy: str             # 사용된 융합 전략
    warnings: List[str] = field(default_factory=list)  # 경고 메시지
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "documents": self.documents,
            "confidence": round(self.confidence, 3),
            "explanation": self.explanation,
            "source_scores": [
                {
                    "source": s.source_name,
                    "raw_score": round(s.raw_score, 3),
                    "normalized_score": round(s.normalized_score, 3),
                    "weight": s.weight,
                    "contribution": round(s.contribution, 3),
                    "confidence_level": s.confidence_level,
                    "explanation": s.explanation
                }
                for s in self.source_scores
            ],
            "fusion_strategy": self.fusion_strategy,
            "warnings": self.warnings,
            "metadata": self.metadata
        }


class ConfidenceFusion:
    """
    다중 소스 신뢰도 융합 엔진

    Features:
    - 가중치 기반 점수 통합
    - 자동 점수 정규화
    - 상충 정보 감지
    - 설명 가능한 결과 생성
    """

    DEFAULT_WEIGHTS = {
        'vector': 0.40,      # 의미적 유사도 (문서 검색)
        'ontology': 0.35,    # 논리적 추론 (규칙 기반)
        'entity': 0.25       # 엔티티 연결 (구조적 관계)
    }

    CONFIDENCE_THRESHOLDS = {
        'high': 0.75,
        'medium': 0.50,
        'low': 0.25
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        normalization: ScoreNormalizationMethod = ScoreNormalizationMethod.MIN_MAX,
        strategy: FusionStrategy = FusionStrategy.WEIGHTED_SUM,
        min_sources: int = 1,  # 최소 필요 소스 수
        conflict_threshold: float = 0.3  # 상충 감지 임계값
    ):
        """
        초기화

        Args:
            weights: 소스별 가중치 (합이 1.0이어야 함)
            normalization: 정규화 방법
            strategy: 융합 전략
            min_sources: 최소 필요 소스 수
            conflict_threshold: 점수 차이가 이보다 크면 상충으로 간주
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        self._validate_weights()

        self.normalization = normalization
        self.strategy = strategy
        self.min_sources = min_sources
        self.conflict_threshold = conflict_threshold

    def _validate_weights(self):
        """가중치 검증 (합이 1.0)"""
        total = sum(self.weights.values())
        if not 0.99 <= total <= 1.01:  # 부동소수점 오차 허용
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    def fuse(
        self,
        vector_results: Optional[List[SearchResult]] = None,
        ontology_results: Optional[List[InferenceResult]] = None,
        entity_links: Optional[List[LinkedEntity]] = None,
        query: Optional[str] = None
    ) -> FusedResult:
        """
        다중 소스 융합

        Args:
            vector_results: 벡터/키워드 검색 결과
            ontology_results: 온톨로지 추론 결과
            entity_links: 엔티티 연결 결과
            query: 원본 쿼리 (설명 생성용)

        Returns:
            FusedResult
        """
        # 1. 소스 수집 및 검증
        available_sources = {}
        if vector_results:
            available_sources['vector'] = vector_results
        if ontology_results:
            available_sources['ontology'] = ontology_results
        if entity_links:
            available_sources['entity'] = entity_links

        if len(available_sources) < self.min_sources:
            return self._create_empty_result(
                f"Insufficient sources: {len(available_sources)} < {self.min_sources}"
            )

        # 2. 각 소스별 점수 계산 및 정규화
        source_scores = []
        raw_scores = {}

        if 'vector' in available_sources:
            score = self._compute_vector_score(vector_results)
            raw_scores['vector'] = score
            source_scores.append(self._create_source_score('vector', score))

        if 'ontology' in available_sources:
            score = self._compute_ontology_score(ontology_results)
            raw_scores['ontology'] = score
            source_scores.append(self._create_source_score('ontology', score))

        if 'entity' in available_sources:
            score = self._compute_entity_score(entity_links)
            raw_scores['entity'] = score
            source_scores.append(self._create_source_score('entity', score))

        # 3. 점수 정규화
        normalized_scores = self._normalize_scores(raw_scores)

        # 4. 가중치 재조정 (없는 소스 제외)
        adjusted_weights = self._adjust_weights(available_sources)

        # 5. 최종 신뢰도 계산
        final_confidence = self._apply_fusion_strategy(
            normalized_scores, adjusted_weights
        )

        # 6. 각 소스별 기여도 계산
        for source_score in source_scores:
            source_name = source_score.source_name
            source_score.normalized_score = normalized_scores[source_name]
            source_score.weight = adjusted_weights[source_name]
            source_score.contribution = (
                normalized_scores[source_name] * adjusted_weights[source_name]
            )

        # 7. 상충 감지
        warnings = self._detect_conflicts(normalized_scores)

        # 8. 문서 통합
        documents = self._merge_documents(
            vector_results, ontology_results, entity_links
        )

        # 9. 설명 생성
        explanation = self._generate_explanation(
            source_scores, final_confidence, query
        )

        return FusedResult(
            documents=documents,
            confidence=final_confidence,
            explanation=explanation,
            source_scores=source_scores,
            fusion_strategy=self.strategy.value,
            warnings=warnings,
            metadata={
                "available_sources": list(available_sources.keys()),
                "adjusted_weights": adjusted_weights,
                "normalization_method": self.normalization.value
            }
        )

    def _compute_vector_score(self, results: List[SearchResult]) -> float:
        """벡터 검색 점수 계산 (최고점 또는 평균)"""
        if not results:
            return 0.0
        scores = [r.score for r in results if 0 <= r.score <= 1]
        if not scores:
            return 0.0
        # Top-K 평균 (상위 3개)
        top_k = sorted(scores, reverse=True)[:3]
        return np.mean(top_k)

    def _compute_ontology_score(self, results: List[InferenceResult]) -> float:
        """온톨로지 추론 점수 계산"""
        if not results:
            return 0.0
        # 가장 높은 신뢰도 추론 사용
        confidences = [r.confidence for r in results if 0 <= r.confidence <= 1]
        if not confidences:
            return 0.0
        return max(confidences)

    def _compute_entity_score(self, entities: List[LinkedEntity]) -> float:
        """엔티티 연결 점수 계산"""
        if not entities:
            return 0.0
        # 연결 강도 평균
        confidences = [e.link_confidence for e in entities if 0 <= e.link_confidence <= 1]
        if not confidences:
            return 0.0
        return np.mean(confidences)

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """점수 정규화"""
        if not scores:
            return {}

        if self.normalization == ScoreNormalizationMethod.NONE:
            return scores

        values = list(scores.values())

        if self.normalization == ScoreNormalizationMethod.MIN_MAX:
            min_val = min(values)
            max_val = max(values)
            if max_val - min_val < 1e-6:  # 모두 같으면
                return {k: 1.0 for k in scores}
            return {
                k: (v - min_val) / (max_val - min_val)
                for k, v in scores.items()
            }

        elif self.normalization == ScoreNormalizationMethod.SOFTMAX:
            exp_values = np.exp(values)
            softmax = exp_values / np.sum(exp_values)
            return dict(zip(scores.keys(), softmax))

        elif self.normalization == ScoreNormalizationMethod.Z_SCORE:
            mean_val = np.mean(values)
            std_val = np.std(values)
            if std_val < 1e-6:
                return {k: 0.5 for k in scores}
            z_scores = {k: (v - mean_val) / std_val for k, v in scores.items()}
            # Z-Score를 0~1로 재스케일 (대략 -3~+3 → 0~1)
            return {k: np.clip((z + 3) / 6, 0, 1) for k, z in z_scores.items()}

        return scores

    def _adjust_weights(self, available_sources: Dict[str, Any]) -> Dict[str, float]:
        """
        가중치 재조정 (없는 소스 제외하고 나머지 재분배)

        Example:
            원래: {vector: 0.4, ontology: 0.35, entity: 0.25}
            entity 없음 → {vector: 0.4/0.75=0.533, ontology: 0.35/0.75=0.467}
        """
        active_weights = {
            k: v for k, v in self.weights.items()
            if k in available_sources
        }
        total = sum(active_weights.values())
        if total < 1e-6:
            # 모든 가중치가 0이면 균등 분배
            return {k: 1.0 / len(active_weights) for k in active_weights}
        return {k: v / total for k, v in active_weights.items()}

    def _apply_fusion_strategy(
        self,
        scores: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """융합 전략 적용"""
        if not scores:
            return 0.0

        values = list(scores.values())
        weight_list = [weights.get(k, 0) for k in scores.keys()]

        if self.strategy == FusionStrategy.WEIGHTED_SUM:
            # 가중합
            return sum(s * w for s, w in zip(values, weight_list))

        elif self.strategy == FusionStrategy.HARMONIC_MEAN:
            # 조화평균 (모든 점수가 높아야 높음 - 보수적)
            if any(v == 0 for v in values):
                return 0.0
            weighted_reciprocals = sum(w / v for v, w in zip(values, weight_list))
            return 1.0 / weighted_reciprocals if weighted_reciprocals > 0 else 0.0

        elif self.strategy == FusionStrategy.GEOMETRIC_MEAN:
            # 기하평균
            product = np.prod([v ** w for v, w in zip(values, weight_list)])
            return product

        elif self.strategy == FusionStrategy.MAX_SCORE:
            # 최대값 (낙관적)
            return max(values)

        elif self.strategy == FusionStrategy.MIN_SCORE:
            # 최소값 (매우 보수적)
            return min(values)

        return 0.0

    def _detect_conflicts(self, scores: Dict[str, float]) -> List[str]:
        """상충되는 정보 감지"""
        warnings = []
        if len(scores) < 2:
            return warnings

        values = list(scores.values())
        max_score = max(values)
        min_score = min(values)

        if max_score - min_score > self.conflict_threshold:
            warnings.append(
                f"점수 불일치 감지: 최대 {max_score:.2f} vs 최소 {min_score:.2f} "
                f"(차이 {max_score - min_score:.2f})"
            )

        return warnings

    def _merge_documents(
        self,
        vector_results: Optional[List[SearchResult]],
        ontology_results: Optional[List[InferenceResult]],
        entity_links: Optional[List[LinkedEntity]]
    ) -> List[Dict[str, Any]]:
        """문서들을 통합"""
        documents = []

        if vector_results:
            for result in vector_results:
                documents.append({
                    "source": "vector",
                    "content": result.content,
                    "score": result.score,
                    "metadata": result.metadata
                })

        if ontology_results:
            for result in ontology_results:
                documents.append({
                    "source": "ontology",
                    "content": result.insight,
                    "score": result.confidence,
                    "evidence": result.evidence,
                    "rule": result.rule_name
                })

        if entity_links:
            for entity in entity_links:
                documents.append({
                    "source": "entity",
                    "entity_id": entity.entity_id,
                    "entity_name": entity.entity_name,
                    "entity_type": entity.entity_type,
                    "score": entity.link_confidence,
                    "context": entity.context,
                    "metadata": entity.metadata
                })

        return documents

    def _create_source_score(self, source_name: str, raw_score: float) -> SourceScore:
        """SourceScore 객체 생성"""
        confidence_level = self._get_confidence_level(raw_score)
        explanation = self._get_source_explanation(source_name, raw_score)

        return SourceScore(
            source_name=source_name,
            raw_score=raw_score,
            normalized_score=0.0,  # 나중에 설정됨
            weight=self.weights.get(source_name, 0.0),
            contribution=0.0,  # 나중에 계산됨
            confidence_level=confidence_level,
            explanation=explanation
        )

    def _get_confidence_level(self, score: float) -> str:
        """신뢰도 수준 결정"""
        if score >= self.CONFIDENCE_THRESHOLDS['high']:
            return 'high'
        elif score >= self.CONFIDENCE_THRESHOLDS['medium']:
            return 'medium'
        else:
            return 'low'

    def _get_source_explanation(self, source: str, score: float) -> str:
        """소스별 설명 생성"""
        level = self._get_confidence_level(score)

        explanations = {
            'vector': {
                'high': f"높은 의미적 유사도 (점수: {score:.2f})",
                'medium': f"중간 수준의 의미적 유사도 (점수: {score:.2f})",
                'low': f"낮은 의미적 유사도 (점수: {score:.2f})"
            },
            'ontology': {
                'high': f"높은 신뢰도의 온톨로지 추론 (점수: {score:.2f})",
                'medium': f"중간 신뢰도의 온톨로지 추론 (점수: {score:.2f})",
                'low': f"낮은 신뢰도의 온톨로지 추론 (점수: {score:.2f})"
            },
            'entity': {
                'high': f"강한 엔티티 연결 (점수: {score:.2f})",
                'medium': f"중간 수준의 엔티티 연결 (점수: {score:.2f})",
                'low': f"약한 엔티티 연결 (점수: {score:.2f})"
            }
        }

        return explanations.get(source, {}).get(level, f"{source} 점수: {score:.2f}")

    def _generate_explanation(
        self,
        source_scores: List[SourceScore],
        final_confidence: float,
        query: Optional[str]
    ) -> str:
        """종합 설명 생성"""
        overall_level = self._get_confidence_level(final_confidence)

        # 기여도 높은 순으로 정렬
        sorted_sources = sorted(
            source_scores, key=lambda s: s.contribution, reverse=True
        )

        explanation_parts = []

        # 전체 신뢰도
        if overall_level == 'high':
            explanation_parts.append(
                f"높은 신뢰도 (최종 점수: {final_confidence:.2f})로 관련성이 높습니다."
            )
        elif overall_level == 'medium':
            explanation_parts.append(
                f"중간 수준의 신뢰도 (최종 점수: {final_confidence:.2f})로 관련성이 있습니다."
            )
        else:
            explanation_parts.append(
                f"낮은 신뢰도 (최종 점수: {final_confidence:.2f})로 관련성이 제한적입니다."
            )

        # 주요 기여 소스 (기여도 > 0.15)
        major_contributors = [s for s in sorted_sources if s.contribution > 0.15]
        if major_contributors:
            contrib_strs = [
                f"{s.source_name}({s.contribution:.2f})"
                for s in major_contributors
            ]
            explanation_parts.append(
                f"주요 근거: {', '.join(contrib_strs)}"
            )

        # 각 소스별 설명
        for source in sorted_sources:
            if source.contribution > 0.05:  # 5% 이상 기여
                explanation_parts.append(f"• {source.explanation}")

        return " ".join(explanation_parts)

    def _create_empty_result(self, reason: str) -> FusedResult:
        """빈 결과 생성 (오류 시)"""
        return FusedResult(
            documents=[],
            confidence=0.0,
            explanation=f"결과를 생성할 수 없습니다: {reason}",
            source_scores=[],
            fusion_strategy=self.strategy.value,
            warnings=[reason],
            metadata={}
        )


# =========================================================================
# 편의 함수
# =========================================================================

def create_default_fusion() -> ConfidenceFusion:
    """기본 설정으로 Fusion 객체 생성"""
    return ConfidenceFusion(
        weights=ConfidenceFusion.DEFAULT_WEIGHTS,
        normalization=ScoreNormalizationMethod.MIN_MAX,
        strategy=FusionStrategy.WEIGHTED_SUM,
        min_sources=1,
        conflict_threshold=0.3
    )


def create_conservative_fusion() -> ConfidenceFusion:
    """보수적 Fusion (Harmonic Mean 사용)"""
    return ConfidenceFusion(
        weights=ConfidenceFusion.DEFAULT_WEIGHTS,
        normalization=ScoreNormalizationMethod.MIN_MAX,
        strategy=FusionStrategy.HARMONIC_MEAN,  # 모든 점수가 높아야 높음
        min_sources=2,  # 최소 2개 소스 필요
        conflict_threshold=0.2  # 더 엄격한 상충 감지
    )


def create_optimistic_fusion() -> ConfidenceFusion:
    """낙관적 Fusion (Max Score 사용)"""
    return ConfidenceFusion(
        weights=ConfidenceFusion.DEFAULT_WEIGHTS,
        normalization=ScoreNormalizationMethod.SOFTMAX,
        strategy=FusionStrategy.MAX_SCORE,  # 최고 점수 사용
        min_sources=1,
        conflict_threshold=0.5
    )
