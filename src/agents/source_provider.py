"""
Source Provider
출처 추출 및 포맷팅 모듈

Perplexity/Liner 스타일 상세 출처 제공:
- 7개 출처 유형 지원
- ASIN 기반 제품 추적
- 마크다운 포맷팅
"""

from datetime import datetime
from typing import Any

from src.rag.hybrid_retriever import HybridContext


class SourceProvider:
    """Extracts and formats source citations for chatbot responses.

    Supports 7 source types and Perplexity-style markdown formatting.
    """

    def __init__(self, config: dict[str, Any] | None = None, knowledge_graph=None):
        """
        Args:
            config: 설정 딕셔너리
            knowledge_graph: KnowledgeGraph 인스턴스 (optional, for category hierarchy)
        """
        self.config = config or {}
        self.knowledge_graph = knowledge_graph

    def extract_sources(
        self,
        hybrid_context: HybridContext,
        current_data: dict[str, Any] | None = None,
        external_signals: list[Any] | None = None,
        model: str = "gpt-4.1-mini",
    ) -> list[dict[str, Any]]:
        """Extract sources from RAG context and response.

        Args:
            hybrid_context: 하이브리드 검색 컨텍스트
            current_data: 현재 데이터 컨텍스트
            external_signals: 외부 신호 리스트 (Tavily 뉴스, RSS, Reddit 등)
            model: LLM 모델명

        Returns:
            출처 정보 리스트 (유형별 상세 정보 포함)
        """
        sources = []

        # 1. 크롤링 데이터 출처 - URL 및 상세 정보 추가 (ASIN 포함)
        if current_data:
            metadata = current_data.get("metadata", {})
            data_date = metadata.get("data_date", "")
            categories = current_data.get("categories", {})

            total_products = (
                sum(len(cat_data.get("rank_records", [])) for cat_data in categories.values())
                if categories
                else 0
            )

            # 질의에서 언급된 제품의 ASIN 추출
            mentioned_asins = self._extract_mentioned_asins(hybrid_context, categories)

            crawled_source = {
                "type": "crawled_data",
                "icon": "📊",
                "description": "Amazon Best Sellers 크롤링 데이터",
                "collected_at": data_date,
                "url": "https://www.amazon.com/gp/bestsellers/beauty",
                "details": {
                    "categories": list(categories.keys()) if categories else [],
                    "total_products": total_products,
                    "snapshot_date": data_date,
                },
            }

            # 관련 제품의 ASIN 정보 추가
            if mentioned_asins:
                crawled_source["mentioned_products"] = mentioned_asins

            sources.append(crawled_source)

        # 2. Knowledge Graph 출처 - 엔티티 및 관계 정보 추가
        if hybrid_context.ontology_facts:
            sources.append(
                {
                    "type": "knowledge_graph",
                    "icon": "🔗",
                    "description": "지식 그래프 관계 데이터",
                    "fact_count": len(hybrid_context.ontology_facts),
                    "entities": self._extract_entity_names(hybrid_context.ontology_facts),
                    "relations": self._extract_relation_types(hybrid_context.ontology_facts),
                    "details": {
                        "source": "Amazon US 실시간 데이터 기반 지식 그래프",
                        "fact_count": len(hybrid_context.ontology_facts),
                    },
                }
            )

        # 3. 온톨로지 추론 출처 - 규칙 상세 정보
        if hybrid_context.inferences:
            for inf in hybrid_context.inferences:
                sources.append(
                    {
                        "type": "ontology_inference",
                        "icon": "🧠",
                        "description": f"온톨로지 규칙: {inf.rule_name}",
                        "rule_name": inf.rule_name,
                        "confidence": inf.confidence,
                        "evidence": inf.evidence,
                        "insight_type": inf.insight_type.value
                        if hasattr(inf.insight_type, "value")
                        else str(inf.insight_type),
                        "details": {"insight": inf.insight, "recommendation": inf.recommendation},
                    }
                )

        # 4. RAG 문서 출처 - 파일 경로 및 관련성 점수
        rag_sources_map = {}
        for chunk in hybrid_context.rag_chunks:
            metadata = chunk.get("metadata", {})
            doc_id = metadata.get("doc_id", "")
            title = metadata.get("title", "")
            file_path = metadata.get("file_path", "")
            score = chunk.get("score", 0)
            section = metadata.get("section", "")

            if doc_id or title:
                doc_key = doc_id or title
                # 같은 문서의 여러 청크 중 가장 높은 점수만 유지
                if doc_key not in rag_sources_map or score > rag_sources_map[doc_key].get(
                    "relevance_score", 0
                ):
                    rag_sources_map[doc_key] = {
                        "type": "rag_document",
                        "icon": "📄",
                        "description": title or doc_id,
                        "file_path": file_path,
                        "section": section,
                        "relevance_score": score,
                        "details": {"doc_id": doc_id, "title": title},
                    }

        sources.extend(rag_sources_map.values())

        # 5. 카테고리 계층 출처
        if (
            hybrid_context.entities
            and hybrid_context.entities.get("categories")
            and self.knowledge_graph
        ):
            for category in hybrid_context.entities["categories"][:3]:  # 최대 3개
                hierarchy = self.knowledge_graph.get_category_hierarchy(category)
                if "error" not in hierarchy:
                    path = []
                    if hierarchy.get("ancestors"):
                        path = [a["name"] for a in reversed(hierarchy["ancestors"])]
                    path.append(hierarchy.get("name", category))

                    sources.append(
                        {
                            "type": "category_hierarchy",
                            "icon": "🗂️",
                            "description": "카테고리 계층 구조",
                            "path": path,
                            "level": hierarchy.get("level", 0),
                            "url": hierarchy.get("url", ""),
                            "details": {"category": category, "hierarchy_depth": len(path)},
                        }
                    )

        # 6. 외부 신호 출처 (Tavily 뉴스, RSS, Reddit 등)
        if external_signals:
            for signal in external_signals[:5]:  # 상위 5개만
                signal_source = getattr(signal, "source", "unknown")
                reliability = 0.7  # 기본값

                # 메타데이터에서 신뢰도 추출
                if hasattr(signal, "metadata") and signal.metadata:
                    reliability = signal.metadata.get("reliability_score", 0.7)

                # 소스 유형에 따라 아이콘 결정
                if "tavily" in signal_source.lower() or "news" in signal_source.lower():
                    icon = "📰"
                    source_type = "external_news"
                elif "reddit" in signal_source.lower():
                    icon = "💬"
                    source_type = "social_media"
                elif "rss" in signal_source.lower():
                    icon = "📡"
                    source_type = "rss_feed"
                elif "youtube" in signal_source.lower():
                    icon = "📺"
                    source_type = "social_media"
                else:
                    icon = "🌐"
                    source_type = "external_source"

                sources.append(
                    {
                        "type": source_type,
                        "icon": icon,
                        "description": getattr(signal, "title", "Unknown"),
                        "source": signal_source,
                        "url": getattr(signal, "url", ""),
                        "published_at": getattr(signal, "published_at", ""),
                        "reliability_score": reliability,
                        "relevance_score": getattr(signal, "relevance_score", 0.5),
                        "details": {
                            "content_preview": getattr(signal, "content", "")[:200]
                            if hasattr(signal, "content")
                            else "",
                            "tier": getattr(signal, "tier", "unknown"),
                        },
                    }
                )

        # 7. AI 모델 출처 (항상 포함)
        sources.append(
            {
                "type": "ai_model",
                "icon": "🤖",
                "description": f"AI 분석: {model}",
                "model": model,
                "disclaimer": "AI가 생성한 분석입니다. 중요한 의사결정 시 추가 검증을 권장합니다.",
                "generated_at": datetime.now().isoformat(),
            }
        )

        return sources

    def format_sources_for_display(self, sources: list[dict[str, Any]]) -> str:
        """Format sources as Perplexity-style markdown with numbered citations.

        Args:
            sources: 출처 정보 리스트

        Returns:
            마크다운 형식의 출처 섹션
        """
        if not sources:
            return ""

        lines = ["\n\n---"]

        # 데이터 출처 시점을 명확히 표시
        crawled_source = next((s for s in sources if s["type"] == "crawled_data"), None)
        if crawled_source:
            collected_at = crawled_source.get("collected_at", "")
            if collected_at:
                lines.append(f"📅 **데이터 기준: Amazon US Best Sellers {collected_at} 수집**")
                lines.append("*(Amazon은 Best Sellers 순위를 매 시간 업데이트합니다)*")
                lines.append("")

        lines.extend(["**📚 출처 및 참고자료:**", ""])

        for i, source in enumerate(sources, 1):
            icon = source.get("icon", "•")
            desc = source.get("description", "알 수 없는 출처")

            if source["type"] == "crawled_data":
                collected = source.get("collected_at", "")
                url = source.get("url", "")
                details = source.get("details", {})
                total = details.get("total_products", 0)
                mentioned_products = source.get("mentioned_products", [])

                lines.append(f"{i}. {icon} **{desc}**")
                lines.append(f"   - 수집일: {collected}")
                if url:
                    lines.append(f"   - URL: {url}")
                if total > 0:
                    lines.append(f"   - 총 제품 수: {total}개")

                # ASIN 기반 제품 추적 정보 표시
                if mentioned_products:
                    lines.append("   - 📦 관련 제품 (ASIN 기준):")
                    for prod in mentioned_products[:3]:  # 최대 3개 표시
                        asin = prod.get("asin", "")
                        name = prod.get("name", "")
                        rank = prod.get("rank", "")
                        category = prod.get("category", "")
                        lines.append(f"     • [{asin}] {name} (#{rank} in {category})")

                lines.append("")

            elif source["type"] == "knowledge_graph":
                fact_count = source.get("fact_count", 0)
                entities = source.get("entities", [])
                relations = source.get("relations", [])
                lines.append(f"{i}. {icon} **{desc}** ({fact_count}개 관계)")
                if entities:
                    lines.append(f"   - 주요 엔티티: {', '.join(entities[:3])}")
                if relations:
                    lines.append(f"   - 관계 유형: {', '.join(relations[:3])}")
                lines.append("")

            elif source["type"] == "ontology_inference":
                conf = source.get("confidence", 0) * 100
                rule_name = source.get("rule_name", "알 수 없음")
                lines.append(f"{i}. {icon} **{desc}**")
                lines.append(f"   - 신뢰도: {conf:.0f}%")
                lines.append(f"   - 규칙: {rule_name}")
                lines.append("")

            elif source["type"] == "rag_document":
                file_path = source.get("file_path", "")
                section = source.get("section", "")
                score = source.get("relevance_score", 0)
                file_name = file_path.split("/")[-1] if file_path else ""
                lines.append(f"{i}. {icon} **{desc}**")
                if file_name:
                    lines.append(f"   - 파일: {file_name}")
                if section:
                    lines.append(f"   - 섹션: {section}")
                if score > 0:
                    lines.append(f"   - 관련도: {score:.2f}")
                lines.append("")

            elif source["type"] == "category_hierarchy":
                path = source.get("path", [])
                level = source.get("level", 0)
                url = source.get("url", "")
                lines.append(f"{i}. {icon} **{desc}**")
                if path:
                    lines.append(f"   - 계층: {' > '.join(path)}")
                lines.append(f"   - 레벨: {level}")
                if url:
                    lines.append(f"   - URL: {url}")
                lines.append("")

            elif source["type"] in ["external_news", "rss_feed"]:
                # 외부 뉴스 / RSS 피드 (Tavily, Allure, WWD 등)
                url = source.get("url", "")
                published_at = source.get("published_at", "")
                reliability = source.get("reliability_score", 0.7) * 100
                source_name = source.get("source", "")
                lines.append(f"{i}. {icon} **{desc}** (신뢰도: {reliability:.0f}%)")
                if source_name:
                    lines.append(f"   - 출처: {source_name}")
                if published_at:
                    lines.append(f"   - 날짜: {published_at}")
                if url:
                    lines.append(f"   - URL: {url}")
                lines.append("")

            elif source["type"] == "social_media":
                # 소셜 미디어 (Reddit, YouTube 등)
                url = source.get("url", "")
                published_at = source.get("published_at", "")
                reliability = source.get("reliability_score", 0.5) * 100
                source_name = source.get("source", "")
                relevance = source.get("relevance_score", 0)
                lines.append(f"{i}. {icon} **{desc}** (신뢰도: {reliability:.0f}%)")
                if source_name:
                    lines.append(f"   - 플랫폼: {source_name}")
                if published_at:
                    lines.append(f"   - 날짜: {published_at}")
                if relevance > 0:
                    lines.append(f"   - 관련도: {relevance:.2f}")
                if url:
                    lines.append(f"   - URL: {url}")
                lines.append("")

            elif source["type"] == "ai_model":
                model = source.get("model", "")
                disclaimer = source.get("disclaimer", "")
                lines.append(f"{i}. {icon} **{desc}**")
                if model:
                    lines.append(f"   - 모델: {model}")
                if disclaimer:
                    lines.append(f"   - 참고: {disclaimer}")
                lines.append("")

        return "\n".join(lines)

    def _extract_entity_names(self, ontology_facts) -> list[str]:
        """KG facts에서 엔티티 이름 추출"""
        entities = set()

        if isinstance(ontology_facts, list):
            for fact in ontology_facts:
                if isinstance(fact, dict):
                    subject = fact.get("subject", "")
                    obj = fact.get("object", "")
                    if subject:
                        entities.add(subject)
                    if obj:
                        entities.add(obj)
        elif isinstance(ontology_facts, dict):
            # 단일 fact인 경우
            subject = ontology_facts.get("subject", "")
            obj = ontology_facts.get("object", "")
            if subject:
                entities.add(subject)
            if obj:
                entities.add(obj)

        # None이나 빈 문자열 제거 후 최대 5개 반환
        return list(filter(None, entities))[:5]

    def _extract_relation_types(self, ontology_facts) -> list[str]:
        """KG facts에서 관계 유형 추출"""
        relations = set()

        if isinstance(ontology_facts, list):
            for fact in ontology_facts:
                if isinstance(fact, dict):
                    predicate = fact.get("predicate", "")
                    if predicate:
                        relations.add(predicate)
        elif isinstance(ontology_facts, dict):
            # 단일 fact인 경우
            predicate = ontology_facts.get("predicate", "")
            if predicate:
                relations.add(predicate)

        # None이나 빈 문자열 제거
        return list(filter(None, relations))

    def _extract_mentioned_asins(
        self, hybrid_context: HybridContext, categories: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """질의에서 언급된 제품의 ASIN 정보 추출"""
        mentioned_products = []
        seen_asins = set()

        # 1. KG 엔티티에서 제품명/브랜드 추출
        mentioned_brands = set()
        if hybrid_context.entities:
            mentioned_brands = set(hybrid_context.entities.get("brands", []))

        # 2. 카테고리 데이터에서 관련 제품 ASIN 추출
        for category_id, cat_data in categories.items():
            rank_records = cat_data.get("rank_records", [])

            for record in rank_records:
                asin = record.get("asin", "")
                brand = record.get("brand", "")
                product_name = record.get("product_name", record.get("title", ""))
                rank = record.get("rank", 0)

                # 이미 처리된 ASIN 스킵
                if asin in seen_asins:
                    continue

                # 언급된 브랜드의 제품만 포함 (최대 5개)
                if brand in mentioned_brands:
                    seen_asins.add(asin)
                    mentioned_products.append(
                        {
                            "asin": asin,
                            "name": product_name,
                            "brand": brand,
                            "rank": rank,
                            "category": category_id,
                            "url": f"https://www.amazon.com/dp/{asin}" if asin else "",
                        }
                    )

                    if len(mentioned_products) >= 5:
                        break

            if len(mentioned_products) >= 5:
                break

        # 순위 기준 정렬
        mentioned_products.sort(key=lambda x: x.get("rank", 999))
        return mentioned_products[:5]
