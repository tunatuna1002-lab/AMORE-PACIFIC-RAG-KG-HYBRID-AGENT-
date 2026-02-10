# src/domain - Clean Architecture Layer 1

## OVERVIEW

Pure domain entities and interface protocols with zero external dependencies.

## ENTITIES

| Entity | File | Purpose |
|--------|------|---------|
| Product | `entities/product.py` | ASIN, title, brand, rank, price |
| RankRecord | `entities/product.py` | Historical rank tracking |
| ProductMetrics | `entities/product.py` | Product-level KPIs |
| Brand | `entities/brand.py` | Brand identity + origin |
| BrandMetrics | `entities/brand.py` | Brand-level KPIs |
| Category | `entities/market.py` | Amazon category hierarchy |
| Snapshot | `entities/market.py` | Point-in-time market state |
| MarketMetrics | `entities/market.py` | Market-level KPIs |
| Relation | `entities/relations.py` | KG triple (subject, predicate, object) |
| InferenceResult | `entities/relations.py` | Reasoner output |

## INTERFACES

| Protocol | File | Implementors |
|----------|------|--------------|
| CrawlerAgentProtocol | `interfaces/agent.py` | CrawlerAgent |
| StorageAgentProtocol | `interfaces/agent.py` | StorageAgent |
| InsightAgentProtocol | `interfaces/agent.py` | HybridInsightAgent |
| ChatAgentProtocol | `interfaces/agent.py` | HybridChatbotAgent |
| ProductRepository | `interfaces/repository.py` | SQLiteProductRepo |
| RetrieverProtocol | `interfaces/retriever.py` | HybridRetriever |
| KnowledgeGraphProtocol | `interfaces/knowledge_graph.py` | KnowledgeGraph |
| LLMClientProtocol | `interfaces/llm_client.py` | LiteLLMClient |

## IMPORT RULES

```
✅ domain imports: nothing (stdlib only)
✅ application imports: domain
✅ infrastructure imports: domain
❌ domain imports: application, infrastructure, adapters
```

## WARNINGS

⚠️ **ProductMetrics** defined in BOTH:
- `entities/product.py` (product-level)
- `entities/market.py` (market-level)

Use explicit imports to avoid collision:
```python
from src.domain.entities.product import ProductMetrics as ProductProductMetrics
from src.domain.entities.market import ProductMetrics as MarketProductMetrics
```

## ANTI-PATTERNS

- **NEVER** import from application/infrastructure in domain
- **NEVER** add external dependencies to entities
- **NEVER** use concrete classes in Protocols (use Protocol types)
- **NEVER** add business logic to entities (keep them pure data)
