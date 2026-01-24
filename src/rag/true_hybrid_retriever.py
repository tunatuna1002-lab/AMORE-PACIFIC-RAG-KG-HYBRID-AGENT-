"""
True Hybrid Retriever
=====================
진정한 RAG-Ontology Hybrid 통합 검색기

아키텍처:
Query
  │
  ├─→ EntityLinker ─→ Linked Entities
  │
  ├─→ OWLReasoner ─→ Ontology-Guided Filters
  │
  ├─→ Query Expansion (LLM)
  │
  ├─→ Vector Search (filtered) ─→ Candidates
  │
  ├─→ Cross-Encoder Reranking
  │
  └─→ Confidence Fusion ─→ Final Results

핵심 차이점 (vs HybridRetriever):
1. 필수 벡터 검색 (키워드 폴백 제거)
2. OWL 추론 (owlready2 기반 진정한 추론)
3. Entity Linking (쿼리 → 온톨로지 개념 매핑)
4. Confidence Fusion (다중 소스 신뢰도 통합)
5. Reranking (Cross-Encoder 기반 재순위화)
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .entity_linker import EntityLinker, LinkedEntity
from .confidence_fusion import ConfidenceFusion, SearchResult, FusedResult
from .chunker import get_semantic_chunker
from .reranker import get_reranker
from .retriever import DocumentRetriever
from src.ontology.owl_reasoner import OWLReasoner

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """
    진정한 하이브리드 검색 결과

    Attributes:
        query: 원본 쿼리
        documents: 최종 검색 결과 (융합 후)
        ontology_context: 온톨로지 추론 결과
        entity_links: 연결된 엔티티
        confidence: 전체 신뢰도
        combined_context: LLM 프롬프트용 통합 컨텍스트
        metadata: 검색 메타데이터
    """
    query: str
    documents: List[Dict[str, Any]] = field(default_factory=list)
    ontology_context: Dict[str, Any] = field(default_factory=dict)
    entity_links: List[LinkedEntity] = field(default_factory=list)
    confidence: float = 0.0
    combined_context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환 (기존 HybridContext 호환)"""
        def get_entity_type(e):
            """entity_type이 Enum이면 .value, 문자열이면 그대로"""
            return e.entity_type.value if hasattr(e.entity_type, 'value') else e.entity_type

        def get_entity_id(e):
            """ontology_id 또는 concept_label 반환"""
            if hasattr(e, 'ontology_id'):
                return e.ontology_id
            return getattr(e, 'concept_label', e.text)

        return {
            "query": self.query,
            "entities": {
                "brands": [get_entity_id(e) for e in self.entity_links if get_entity_type(e) == "brand"],
                "categories": [get_entity_id(e) for e in self.entity_links if get_entity_type(e) == "category"],
                "indicators": [get_entity_id(e) for e in self.entity_links if get_entity_type(e) == "indicator"],
                "products": [get_entity_id(e) for e in self.entity_links if get_entity_type(e) == "product"],
            },
            "ontology_facts": self.ontology_context.get("facts", []),
            "inferences": self.ontology_context.get("inferences", []),
            "rag_chunks": self.documents,
            "combined_context": self.combined_context,
            "metadata": self.metadata
        }


class TrueHybridRetriever:
    """
    진정한 RAG-Ontology 하이브리드 검색기

    통합 컴포넌트:
    - EntityLinker: 엔티티 추출 및 온톨로지 연결
    - OWLReasoner: OWL 2 기반 형식 추론
    - DocumentRetriever: 벡터 검색 (필수)
    - CrossEncoderReranker: 재순위화
    - ConfidenceFusion: 신뢰도 융합

    사용 예:
        retriever = TrueHybridRetriever()
        await retriever.initialize()

        result = await retriever.retrieve(
            query="LANEIGE Lip Care 경쟁력 분석",
            current_metrics=dashboard_data
        )

        print(result.confidence)
        print(result.combined_context)
    """

    def __init__(
        self,
        knowledge_graph: Optional[Any] = None,
        owl_reasoner: Optional[OWLReasoner] = None,
        doc_retriever: Optional[DocumentRetriever] = None,
        use_semantic_chunking: bool = True,
        use_reranking: bool = True,
        use_query_expansion: bool = True,
        # fusion_method deprecated, using ConfidenceFusion defaults
    ):
        """
        Args:
            knowledge_graph: KnowledgeGraph 인스턴스
            owl_reasoner: OWLReasoner 인스턴스
            doc_retriever: DocumentRetriever 인스턴스
            use_semantic_chunking: Semantic Chunking 사용 여부
            use_reranking: Cross-Encoder Reranking 사용 여부
            use_query_expansion: Query Expansion 사용 여부
            fusion_method: 융합 방법 ("weighted", "rrf", "hybrid")
        """
        # 컴포넌트 초기화
        self.kg = knowledge_graph
        self.owl_reasoner = owl_reasoner or OWLReasoner()

        self.doc_retriever = doc_retriever or DocumentRetriever(
            use_semantic_chunking=use_semantic_chunking,
            use_reranker=use_reranking,
            use_query_expansion=use_query_expansion
        )

        # Entity Linker
        self.entity_linker = EntityLinker(knowledge_graph=self.kg)

        # Confidence Fusion
        self.confidence_fusion = ConfidenceFusion()

        # Reranker (lazy loading)
        self._reranker = None
        self.use_reranking = use_reranking

        # 초기화 상태
        self._initialized = False

    async def initialize(self) -> None:
        """비동기 초기화"""
        if not self._initialized:
            # DocumentRetriever 초기화 (필수)
            await self.doc_retriever.initialize()

            # OWLReasoner 초기화
            await self.owl_reasoner.initialize()

            # Knowledge Graph 데이터를 OWL로 마이그레이션 (옵션)
            if self.kg:
                try:
                    count = self.owl_reasoner.import_from_knowledge_graph(self.kg)
                    if count > 0:
                        logger.info(f"Imported {count} entities to OWL ontology")

                        # 추론 실행
                        self.owl_reasoner.run_reasoner()
                        self.owl_reasoner.infer_market_positions()
                except Exception as e:
                    logger.warning(f"Failed to migrate KG to OWL: {e}")

            self._initialized = True
            logger.info("TrueHybridRetriever initialized")

    async def retrieve(
        self,
        query: str,
        current_metrics: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> HybridResult:
        """
        전체 하이브리드 검색 파이프라인 실행

        Args:
            query: 사용자 쿼리
            current_metrics: 현재 계산된 지표 데이터
            top_k: 반환할 결과 수

        Returns:
            HybridResult
        """
        # 초기화 확인
        if not self._initialized:
            await self.initialize()

        start_time = datetime.now()

        # 결과 객체 초기화
        result = HybridResult(query=query)

        try:
            # 1. Entity Linking
            entities = self._link_entities(query)
            result.entity_links = entities
            logger.debug(f"Linked {len(entities)} entities")

            # 엔티티 신뢰도 평균
            entity_confidence = sum(e.confidence for e in entities) / len(entities) if entities else 0.5

            # 2. Ontology-Guided Filters 생성
            ontology_filters = self._build_ontology_filters(entities)
            logger.debug(f"Ontology filters: {ontology_filters}")

            # 3. Query Expansion (옵션)
            expanded_queries = await self._expand_query(query)

            # 4. Ontology-Guided Vector Search
            vector_results = await self._ontology_guided_search(
                expanded_queries,
                ontology_filters,
                top_k * 3  # 재순위화를 위해 더 많은 후보 검색
            )

            # 5. OWL Ontology Reasoning
            ontology_context = await self._infer_with_ontology(entities, current_metrics)
            result.ontology_context = ontology_context

            # 6. Cross-Encoder Reranking (옵션)
            if self.use_reranking and vector_results:
                reranked_results = await self._rerank(query, vector_results, top_k * 2)
            else:
                reranked_results = vector_results[:top_k * 2]

            # 7. Confidence Fusion (Vector + Ontology + Reranker)
            fused_docs = self._fuse_results(
                vector_results=vector_results[:top_k * 2],
                ontology_results=ontology_context.get("related_docs", []),
                reranked_results=reranked_results,
                entity_confidence=entity_confidence
            )

            # 최종 결과 (top_k)
            # fused_docs는 FusedResult 객체 (documents는 List[Dict])
            fused_documents = fused_docs.documents if hasattr(fused_docs, 'documents') else []
            result.documents = [
                {
                    "id": doc.get("id", str(i)),
                    "content": doc.get("content", ""),
                    "metadata": doc.get("metadata", {}),
                    "score": doc.get("score", 0.5),
                    "rank": i + 1
                }
                for i, doc in enumerate(fused_documents[:top_k])
            ]
            result.metadata["fusion_confidence"] = fused_docs.confidence if hasattr(fused_docs, 'confidence') else 0.5

            # 8. Combined Context 생성
            result.combined_context = self._build_combined_context(result)

            # 9. 전체 신뢰도 계산
            # fused_docs는 FusedResult 객체이므로 confidence 속성 사용
            avg_doc_score = fused_docs.confidence if hasattr(fused_docs, 'confidence') else 0.5
            result.confidence = self._calculate_overall_confidence(
                entity_confidence=entity_confidence,
                avg_doc_score=avg_doc_score,
                ontology_coverage=len(ontology_context.get("inferences", [])) / 5  # 최대 5개 인사이트 가정
            )

            # 메타데이터
            result.metadata = {
                "retrieval_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "entity_count": len(entities),
                "entity_confidence": entity_confidence,
                "vector_results_count": len(vector_results),
                "reranked_results_count": len(reranked_results) if self.use_reranking else 0,
                "final_results_count": len(result.documents),
                "ontology_inferences_count": len(ontology_context.get("inferences", [])),
                "query_expanded": len(expanded_queries) > 1
            }

        except Exception as e:
            logger.error(f"True hybrid retrieval failed: {e}")
            result.metadata["error"] = str(e)
            result.confidence = 0.0

        return result

    def _link_entities(self, query: str) -> List[LinkedEntity]:
        """
        쿼리에서 엔티티 추출 및 온톨로지 연결

        Args:
            query: 사용자 쿼리

        Returns:
            LinkedEntity 리스트
        """
        return self.entity_linker.link(query)

    def _build_ontology_filters(self, entities: List[LinkedEntity]) -> Dict[str, Any]:
        """
        엔티티 기반 벡터 검색 필터 생성

        Args:
            entities: 연결된 엔티티

        Returns:
            벡터 검색 필터 (ChromaDB where 조건)
        """
        return self.entity_linker.get_ontology_filters(entities)

    async def _expand_query(self, query: str) -> List[str]:
        """
        LLM 기반 쿼리 확장

        Args:
            query: 원본 쿼리

        Returns:
            확장된 쿼리 리스트
        """
        return await self.doc_retriever.expand_query(query)

    async def _ontology_guided_search(
        self,
        queries: List[str],
        filters: Dict[str, Any],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        온톨로지 가이드 벡터 검색

        Args:
            queries: 검색 쿼리 리스트
            filters: 온톨로지 필터
            top_k: 반환할 결과 수

        Returns:
            검색 결과 리스트
        """
        all_results = []
        seen_ids = set()

        for q in queries:
            results = await self.doc_retriever.search(
                query=q,
                top_k=top_k // len(queries),
                use_query_expansion=False,  # 이미 확장됨
                use_reranking=False  # 나중에 일괄 재순위화
            )

            # 중복 제거
            for result in results:
                if result["id"] not in seen_ids:
                    # 온톨로지 필터 적용 (메타데이터 매칭)
                    if self._matches_filters(result.get("metadata", {}), filters):
                        all_results.append(result)
                        seen_ids.add(result["id"])

        return all_results[:top_k]

    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """메타데이터가 필터 조건을 만족하는지 확인"""
        if not filters:
            return True

        for key, condition in filters.items():
            meta_value = metadata.get(key)

            if isinstance(condition, dict):
                # $in 연산자
                if "$in" in condition:
                    if meta_value not in condition["$in"]:
                        return False
            else:
                # 직접 비교
                if meta_value != condition:
                    return False

        return True

    async def _infer_with_ontology(
        self,
        entities: List[LinkedEntity],
        current_metrics: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        OWL 온톨로지 추론 실행

        Args:
            entities: 연결된 엔티티
            current_metrics: 현재 메트릭 데이터

        Returns:
            온톨로지 컨텍스트 (추론 결과, 인사이트 등)
        """
        context = {
            "inferences": [],
            "facts": [],
            "related_docs": []
        }

        try:
            # OWL 추론 실행
            inferred_facts = self.owl_reasoner.get_inferred_facts()
            context["facts"] = inferred_facts

            # 브랜드 관련 인사이트 생성
            for entity in entities:
                entity_type = entity.entity_type.value if hasattr(entity.entity_type, 'value') else entity.entity_type
                # entity_id: ontology_id 또는 concept_label 사용
                entity_id = entity.ontology_id if hasattr(entity, 'ontology_id') else getattr(entity, 'concept_label', entity.text)
                if entity_type == "brand":
                    brand_info = self.owl_reasoner.get_brand_info(entity_id)
                    if brand_info:
                        # 시장 포지션 추론
                        position = brand_info.get("market_position")
                        if position:
                            context["inferences"].append({
                                "type": "market_position",
                                "brand": entity_id,
                                "position": position,
                                "sos": brand_info.get("sos", 0.0)
                            })

                        # 경쟁사 정보
                        competitors = brand_info.get("competitors", [])
                        if competitors:
                            context["inferences"].append({
                                "type": "competition",
                                "brand": entity_id,
                                "competitors": competitors[:5]
                            })

        except Exception as e:
            logger.warning(f"Ontology inference failed: {e}")

        return context

    async def _rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Cross-Encoder 기반 재순위화

        Args:
            query: 검색 쿼리
            documents: 문서 리스트
            top_k: 반환할 결과 수

        Returns:
            재순위화된 문서 리스트
        """
        if self._reranker is None:
            self._reranker = get_reranker()

        try:
            ranked_docs = self._reranker.rerank(query, documents, top_k=top_k)

            return [
                {
                    "id": doc.metadata.get("chunk_id", ""),
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "score": doc.score
                }
                for doc in ranked_docs
            ]
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return documents[:top_k]

    def _fuse_results(
        self,
        vector_results: List[Dict[str, Any]],
        ontology_results: List[Dict[str, Any]],
        reranked_results: List[Dict[str, Any]],
        entity_confidence: float
    ) -> List[SearchResult]:
        """
        Confidence Fusion으로 결과 통합

        Args:
            vector_results: 벡터 검색 결과
            ontology_results: 온톨로지 검색 결과
            reranked_results: 재순위화 결과
            entity_confidence: 엔티티 신뢰도

        Returns:
            융합된 SearchResult 리스트
        """
        # ConfidenceFusion API에 맞게 변환
        # vector_results를 SearchResult로 변환
        from src.rag.confidence_fusion import SearchResult as FusionSearchResult
        from src.rag.confidence_fusion import InferenceResult, LinkedEntity as FusionLinkedEntity

        fusion_vector = [
            FusionSearchResult(
                content=r.get("content", ""),
                score=r.get("score", 0.5),
                metadata=r.get("metadata", {}),
                source="vector"
            )
            for r in vector_results
        ] if vector_results else None

        fusion_ontology = [
            InferenceResult(
                insight=r.get("insight", ""),
                confidence=r.get("confidence", 0.5),
                evidence=r.get("evidence", {}),
                rule_name=r.get("rule_name")
            )
            for r in ontology_results
        ] if ontology_results else None

        # reranked_results를 entity_links로 변환
        fusion_entities = [
            FusionLinkedEntity(
                entity_id=str(i),
                entity_name=r.get("text", ""),
                entity_type=r.get("type", "unknown"),
                link_confidence=entity_confidence,
                context=r.get("context", ""),
                metadata=r.get("metadata", {})
            )
            for i, r in enumerate(reranked_results)
        ] if reranked_results else None

        return self.confidence_fusion.fuse(
            vector_results=fusion_vector,
            ontology_results=fusion_ontology,
            entity_links=fusion_entities,
            query=None
        )

    def _build_combined_context(self, result: HybridResult) -> str:
        """
        LLM 프롬프트용 통합 컨텍스트 생성

        Args:
            result: HybridResult

        Returns:
            통합 컨텍스트 문자열
        """
        parts = []

        # 1. 엔티티 정보
        if result.entity_links:
            parts.append("## 추출된 엔티티\n")
            for entity in result.entity_links[:5]:
                # entity_type이 Enum이면 .value, 문자열이면 그대로 사용
                entity_type = entity.entity_type.value if hasattr(entity.entity_type, 'value') else entity.entity_type
                parts.append(f"- {entity_type}: {entity.text} (신뢰도: {entity.confidence:.2f})")
            parts.append("")

        # 2. 온톨로지 추론 결과
        if result.ontology_context.get("inferences"):
            parts.append("## 온톨로지 추론 결과\n")
            for inference in result.ontology_context["inferences"][:3]:
                inf_type = inference.get("type", "")
                if inf_type == "market_position":
                    brand = inference.get("brand", "")
                    position = inference.get("position", "")
                    sos = inference.get("sos", 0.0)
                    parts.append(f"- {brand}의 시장 포지션: {position} (SoS: {sos:.2%})")
                elif inf_type == "competition":
                    brand = inference.get("brand", "")
                    competitors = inference.get("competitors", [])
                    parts.append(f"- {brand}의 경쟁사: {', '.join(competitors[:3])}")
            parts.append("")

        # 3. 검색된 문서
        if result.documents:
            parts.append("## 관련 문서\n")
            for i, doc in enumerate(result.documents[:3], 1):
                title = doc.get("metadata", {}).get("title", "")
                content = doc.get("content", "")

                if title:
                    parts.append(f"### {i}. {title}")

                # 내용 축약 (500자)
                if len(content) > 500:
                    content = content[:500] + "..."

                parts.append(content)
                parts.append("")

        return "\n".join(parts)

    def _calculate_overall_confidence(
        self,
        entity_confidence: float,
        avg_doc_score: float,
        ontology_coverage: float
    ) -> float:
        """
        전체 신뢰도 계산

        Args:
            entity_confidence: 엔티티 연결 신뢰도
            avg_doc_score: 문서 평균 점수
            ontology_coverage: 온톨로지 커버리지

        Returns:
            전체 신뢰도 (0.0 ~ 1.0)
        """
        # 가중 평균
        confidence = (
            0.4 * entity_confidence +
            0.4 * avg_doc_score +
            0.2 * min(ontology_coverage, 1.0)
        )

        return min(max(confidence, 0.0), 1.0)

    def get_stats(self) -> Dict[str, Any]:
        """검색기 통계"""
        return {
            "owl_reasoner": self.owl_reasoner.get_stats() if self.owl_reasoner else {},
            "doc_retriever": {
                "documents_count": len(self.doc_retriever.documents),
                "chunks_count": len(self.doc_retriever.chunks)
            },
            "initialized": self._initialized
        }


# 싱글톤 인스턴스
_retriever_instance: Optional[TrueHybridRetriever] = None


def get_true_hybrid_retriever(
    owl_reasoner: Optional[OWLReasoner] = None,
    knowledge_graph: Optional[Any] = None,
    docs_path: str = "./docs"
) -> TrueHybridRetriever:
    """TrueHybridRetriever 싱글톤 인스턴스 반환

    Args:
        owl_reasoner: OWLReasoner 인스턴스
        knowledge_graph: KnowledgeGraph 인스턴스
        docs_path: RAG 문서 디렉토리 경로

    Returns:
        TrueHybridRetriever 인스턴스
    """
    global _retriever_instance
    if _retriever_instance is None:
        doc_retriever = DocumentRetriever(
            docs_path=docs_path,
            use_semantic_chunking=True,
            use_reranker=True,
            use_query_expansion=True
        )
        _retriever_instance = TrueHybridRetriever(
            owl_reasoner=owl_reasoner,
            knowledge_graph=knowledge_graph,
            doc_retriever=doc_retriever
        )
    return _retriever_instance
