"""
OWL Reasoner (owlready2 기반)
==============================
OWL 2 온톨로지 기반 형식 추론 엔진

## 핵심 개념
- **OWL Ontology**: W3C 표준 온톨로지 언어
- **Reasoner**: Pellet/HermiT 기반 자동 추론
- **Classes**: Brand, Product, Category, Trend
- **Properties**: hasBrand, belongsToCategory, competsWith
- **Restrictions**: SoS 기반 MarketPosition 자동 분류

## 기능
1. OWL 파일 로드/저장
2. 엔티티 관리 (Brand, Product, Category, Trend)
3. Object Properties (관계) 정의
4. 추론기 실행 (Pellet/HermiT)
5. SPARQL 쿼리 지원
6. 기존 KnowledgeGraph 데이터 마이그레이션

## 사용 예
```python
from src.ontology.owl_reasoner import OWLReasoner

reasoner = OWLReasoner()
await reasoner.initialize()

# 브랜드 추가
reasoner.add_brand("LANEIGE", sos=0.25, avg_rank=15.5)
reasoner.add_brand("COSRX", sos=0.18, avg_rank=12.3)

# 경쟁 관계 설정
reasoner.add_competitor_relation("LANEIGE", "COSRX")

# 추론 실행
reasoner.run_reasoner()

# 추론 결과 조회
position = reasoner.get_brand_market_position("LANEIGE")
```

## OWL 클래스 구조
```
owl:Thing
├── Brand
│   ├── DominantBrand (SoS >= 30%)
│   ├── StrongBrand (15% <= SoS < 30%)
│   └── NicheBrand (SoS < 15%)
├── Product
├── Category
└── Trend
```

## Object Properties
- hasBrand: Product → Brand
- hasProduct: Brand → Product (역관계)
- belongsToCategory: Product → Category
- competsWith: Brand → Brand (대칭 관계)
- hasTrend: Brand → Trend

## Data Properties
- shareOfShelf: Brand → float
- averageRank: Brand → float
- productCount: Brand → int
- rank: Product → int
- price: Product → float
- rating: Product → float
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# owlready2 선택적 import
try:
    from owlready2 import (
        AllDisjoint,
        ConstrainedDatatype,
        DataProperty,
        FunctionalProperty,
        ObjectProperty,
        SymmetricProperty,
        Thing,
        destroy_entity,
        get_ontology,
        sync_reasoner_hermit,
        sync_reasoner_pellet,
    )

    OWLREADY2_AVAILABLE = True
    logger.info("owlready2 is available")
except ImportError:
    OWLREADY2_AVAILABLE = False
    logger.warning("owlready2 not installed. OWLReasoner will use fallback mode.")


class OWLReasoner:
    """
    OWL 2 기반 온톨로지 추론 엔진

    owlready2를 사용하여 형식적 온톨로지 추론을 수행합니다.
    owlready2가 없을 경우 기존 OntologyReasoner로 fallback합니다.

    Attributes:
        onto: owlready2 Ontology 객체
        reasoner_type: "pellet" 또는 "hermit"
        owl_file: OWL 파일 경로
    """

    def __init__(
        self,
        owl_file: str | None = None,
        reasoner_type: str = "pellet",
        fallback_reasoner: Any | None = None,
    ):
        """
        Args:
            owl_file: OWL 파일 경로 (None이면 메모리 온톨로지)
            reasoner_type: "pellet" 또는 "hermit"
            fallback_reasoner: owlready2 미설치 시 사용할 OntologyReasoner
        """
        self.owl_file = Path(owl_file) if owl_file else None
        self.reasoner_type = reasoner_type
        self.fallback_reasoner = fallback_reasoner

        # owlready2 사용 불가 시 fallback
        if not OWLREADY2_AVAILABLE:
            logger.warning("Using fallback reasoner due to missing owlready2")
            if not self.fallback_reasoner:
                from .reasoner import OntologyReasoner

                self.fallback_reasoner = OntologyReasoner()
            self.onto = None
            return

        # OWL 온톨로지 초기화
        if self.owl_file and self.owl_file.exists():
            self.onto = get_ontology(f"file://{self.owl_file.absolute()}").load()
            logger.info(f"Loaded ontology from {self.owl_file}")
        else:
            self.onto = get_ontology("http://amorepacific.com/ontology/amore_brand.owl")
            logger.info("Created new in-memory ontology")

        # 네임스페이스 설정
        with self.onto:
            self._define_ontology_structure()

    async def initialize(self):
        """비동기 초기화 (호환성)"""
        logger.info("OWLReasoner initialized")

    # =========================================================================
    # 온톨로지 구조 정의
    # =========================================================================

    def _define_ontology_structure(self):
        """OWL 클래스 및 프로퍼티 정의"""
        if not OWLREADY2_AVAILABLE or not self.onto:
            return

        with self.onto:
            # ===== Classes =====
            # owlready2에서 클래스는 먼저 기존 여부를 확인하고 없으면 생성

            # Brand 클래스
            self.onto.Brand or type("Brand", (Thing,), {"namespace": self.onto})

            # Product 클래스
            self.onto.Product or type("Product", (Thing,), {"namespace": self.onto})

            # Category 클래스
            self.onto.Category or type("Category", (Thing,), {"namespace": self.onto})

            # Trend 클래스
            self.onto.Trend or type("Trend", (Thing,), {"namespace": self.onto})

        # 클래스 정의 후 서브클래스와 프로퍼티 정의
        with self.onto:
            # Market Position 서브클래스
            if not self.onto.DominantBrand:
                type("DominantBrand", (self.onto.Brand,), {"namespace": self.onto})

            if not self.onto.StrongBrand:
                type("StrongBrand", (self.onto.Brand,), {"namespace": self.onto})

            if not self.onto.NicheBrand:
                type("NicheBrand", (self.onto.Brand,), {"namespace": self.onto})

            # ===== A-3: Disjointness Axiom =====
            # Brand 서브클래스 간 상호 배타 (동시에 두 포지션에 속할 수 없음)
            AllDisjoint(
                [
                    self.onto.DominantBrand,
                    self.onto.StrongBrand,
                    self.onto.NicheBrand,
                ]
            )

            # ===== Object Properties =====

            # hasBrand: Product → Brand
            if not self.onto.hasBrand:

                class hasBrand(ObjectProperty):
                    namespace = self.onto
                    domain = [self.onto.Product]
                    range = [self.onto.Brand]
                    python_name = "has_brand"

            # hasProduct: Brand → Product (역관계)
            if not self.onto.hasProduct:

                class hasProduct(ObjectProperty):
                    namespace = self.onto
                    domain = [self.onto.Brand]
                    range = [self.onto.Product]
                    python_name = "has_product"

            # belongsToCategory: Product → Category
            if not self.onto.belongsToCategory:

                class belongsToCategory(ObjectProperty):
                    namespace = self.onto
                    domain = [self.onto.Product]
                    range = [self.onto.Category]
                    python_name = "belongs_to_category"

            # Cardinality: Product must belong to exactly 1 Category
            self.onto.Product.is_a.append(
                self.onto.belongsToCategory.exactly(1, self.onto.Category)
            )

            # competsWith: Brand → Brand (대칭 관계)
            if not self.onto.competsWith:

                class competsWith(SymmetricProperty, ObjectProperty):
                    namespace = self.onto
                    domain = [self.onto.Brand]
                    range = [self.onto.Brand]
                    python_name = "competes_with"

            # hasTrend: Brand → Trend
            if not self.onto.hasTrend:

                class hasTrend(ObjectProperty):
                    namespace = self.onto
                    domain = [self.onto.Brand]
                    range = [self.onto.Trend]
                    python_name = "has_trend"

            # ===== Data Properties =====

            # shareOfShelf: Brand → float
            if not self.onto.shareOfShelf:

                class shareOfShelf(DataProperty, FunctionalProperty):
                    namespace = self.onto
                    domain = [self.onto.Brand]
                    range = [float]
                    python_name = "share_of_shelf"

            # averageRank: Brand → float
            if not self.onto.averageRank:

                class averageRank(DataProperty, FunctionalProperty):
                    namespace = self.onto
                    domain = [self.onto.Brand]
                    range = [float]
                    python_name = "average_rank"

            # productCount: Brand → int
            if not self.onto.productCount:

                class productCount(DataProperty, FunctionalProperty):
                    namespace = self.onto
                    domain = [self.onto.Brand]
                    range = [int]
                    python_name = "product_count"

            # rank: Product → int
            if not self.onto.rank:

                class rank(DataProperty, FunctionalProperty):
                    namespace = self.onto
                    domain = [self.onto.Product]
                    range = [int]
                    python_name = "rank_value"

            # price: Product → float
            if not self.onto.price:

                class price(DataProperty):
                    namespace = self.onto
                    domain = [self.onto.Product]
                    range = [float]
                    python_name = "price_value"

            # rating: Product → float
            if not self.onto.rating:

                class rating(DataProperty):
                    namespace = self.onto
                    domain = [self.onto.Product]
                    range = [float]
                    python_name = "rating_value"

        # ===== A-1: OWL Class Restrictions =====
        # equivalent_to를 사용한 형식적 OWL 2 정의
        with self.onto:
            # DominantBrand ≡ Brand ⊓ ∃shareOfShelf[≥0.30]
            self.onto.DominantBrand.equivalent_to = [
                self.onto.Brand
                & self.onto.shareOfShelf.some(ConstrainedDatatype(float, min_inclusive=0.30))
            ]

            # StrongBrand ≡ Brand ⊓ ∃shareOfShelf[≥0.15 ∧ <0.30]
            self.onto.StrongBrand.equivalent_to = [
                self.onto.Brand
                & self.onto.shareOfShelf.some(
                    ConstrainedDatatype(float, min_inclusive=0.15, max_exclusive=0.30)
                )
            ]

            # NicheBrand ≡ Brand ⊓ ∃shareOfShelf[<0.15]
            self.onto.NicheBrand.equivalent_to = [
                self.onto.Brand
                & self.onto.shareOfShelf.some(ConstrainedDatatype(float, max_exclusive=0.15))
            ]

        # ===== A-2: inverseOf 선언 =====
        # hasBrand ↔ hasProduct 역관계 연결
        with self.onto:
            if self.onto.hasProduct and self.onto.hasBrand:
                self.onto.hasProduct.inverse_property = self.onto.hasBrand

        logger.info("OWL ontology structure defined")

    # =========================================================================
    # 엔티티 추가
    # =========================================================================

    def add_brand(
        self, name: str, sos: float = 0.0, avg_rank: float | None = None, product_count: int = 0
    ) -> bool:
        """
        브랜드 추가

        Args:
            name: 브랜드명
            sos: Share of Shelf (0.0 ~ 1.0)
            avg_rank: 평균 순위
            product_count: 제품 수

        Returns:
            성공 여부
        """
        if not OWLREADY2_AVAILABLE:
            logger.warning("owlready2 not available, skipping add_brand")
            return False

        try:
            with self.onto:
                # 브랜드 인스턴스 생성
                brand_id = name.replace(" ", "_")
                brand = self.onto.Brand(brand_id)

                # Data Properties 설정 (FunctionalProperty는 단일 값)
                brand.share_of_shelf = sos
                if avg_rank is not None:
                    brand.average_rank = avg_rank
                brand.product_count = product_count

                logger.info(f"Added brand: {name} (SoS: {sos:.2%})")
                return True
        except Exception as e:
            logger.error(f"Failed to add brand {name}: {e}")
            return False

    def add_product(
        self,
        asin: str,
        brand: str,
        category: str,
        rank: int,
        price: float | None = None,
        rating: float | None = None,
    ) -> bool:
        """
        제품 추가

        Args:
            asin: Amazon ASIN
            brand: 브랜드명
            category: 카테고리 ID
            rank: 순위
            price: 가격
            rating: 평점

        Returns:
            성공 여부
        """
        if not OWLREADY2_AVAILABLE:
            return False

        try:
            with self.onto:
                # Product 인스턴스 생성
                product = self.onto.Product(asin)
                product.rank_value = rank  # FunctionalProperty
                if price:
                    product.price_value = price
                if rating:
                    product.rating_value = rating

                # Brand 관계
                brand_id = brand.replace(" ", "_")
                brand_instance = self.onto.Brand(brand_id)
                product.has_brand = brand_instance

                # Category 관계
                cat_instance = self.onto.Category(category)
                product.belongs_to_category = cat_instance

                logger.debug(f"Added product: {asin} (brand: {brand}, rank: {rank})")
                return True
        except Exception as e:
            logger.error(f"Failed to add product {asin}: {e}")
            return False

    def add_competitor_relation(self, brand1: str, brand2: str) -> bool:
        """
        경쟁 관계 추가 (대칭 관계)

        Args:
            brand1: 브랜드 1
            brand2: 브랜드 2

        Returns:
            성공 여부
        """
        if not OWLREADY2_AVAILABLE:
            return False

        try:
            with self.onto:
                brand1_id = brand1.replace(" ", "_")
                brand2_id = brand2.replace(" ", "_")

                b1 = self.onto.Brand(brand1_id)
                b2 = self.onto.Brand(brand2_id)

                # competsWith는 대칭 관계이므로 한 방향만 설정
                if b2 not in b1.competes_with:
                    b1.competes_with.append(b2)

                logger.debug(f"Added competitor relation: {brand1} ↔ {brand2}")
                return True
        except Exception as e:
            logger.error(f"Failed to add competitor relation {brand1}-{brand2}: {e}")
            return False

    def add_trend(self, keyword: str, brands: list[str]) -> bool:
        """
        트렌드 키워드 추가

        Args:
            keyword: 트렌드 키워드
            brands: 관련 브랜드 리스트

        Returns:
            성공 여부
        """
        if not OWLREADY2_AVAILABLE:
            return False

        try:
            with self.onto:
                trend = self.onto.Trend(keyword.replace(" ", "_"))

                for brand_name in brands:
                    brand_id = brand_name.replace(" ", "_")
                    brand = self.onto.Brand(brand_id)
                    brand.has_trend.append(trend)

                logger.debug(f"Added trend: {keyword} (brands: {brands})")
                return True
        except Exception as e:
            logger.error(f"Failed to add trend {keyword}: {e}")
            return False

    # =========================================================================
    # 추론 실행
    # =========================================================================

    def run_reasoner(self) -> bool:
        """
        추론기 실행 (Pellet 또는 HermiT)

        Returns:
            성공 여부
        """
        if not OWLREADY2_AVAILABLE:
            logger.warning("Cannot run reasoner without owlready2")
            return False

        try:
            logger.info(f"Running {self.reasoner_type} reasoner...")

            with self.onto:
                if self.reasoner_type.lower() == "pellet":
                    sync_reasoner_pellet(infer_property_values=True, debug=False)
                elif self.reasoner_type.lower() == "hermit":
                    sync_reasoner_hermit(infer_property_values=True, debug=False)
                else:
                    logger.error(f"Unknown reasoner type: {self.reasoner_type}")
                    return False

            logger.info("Reasoner execution completed")
            return True
        except Exception as e:
            logger.error(f"Reasoner execution failed: {e}")
            return False

    def infer_market_positions(self) -> dict[str, str]:
        """
        SoS 기반 시장 포지션 추론

        수동으로 브랜드를 서브클래스에 할당합니다.
        (OWL 제약 조건 대신 Python 로직 사용)

        Returns:
            {brand_name: position}
        """
        if not OWLREADY2_AVAILABLE:
            return {}

        positions = {}

        try:
            with self.onto:
                for brand in self.onto.Brand.instances():
                    # FunctionalProperty는 단일 값 반환
                    sos = brand.share_of_shelf
                    if sos is None:
                        continue

                    # 기존 서브클래스 제거
                    if self.onto.DominantBrand in brand.is_a:
                        brand.is_a.remove(self.onto.DominantBrand)
                    if self.onto.StrongBrand in brand.is_a:
                        brand.is_a.remove(self.onto.StrongBrand)
                    if self.onto.NicheBrand in brand.is_a:
                        brand.is_a.remove(self.onto.NicheBrand)

                    # SoS 기반 분류
                    if sos >= 0.30:
                        brand.is_a.append(self.onto.DominantBrand)
                        positions[brand.name] = "DominantBrand"
                    elif sos >= 0.15:
                        brand.is_a.append(self.onto.StrongBrand)
                        positions[brand.name] = "StrongBrand"
                    else:
                        brand.is_a.append(self.onto.NicheBrand)
                        positions[brand.name] = "NicheBrand"

            logger.info(f"Inferred market positions for {len(positions)} brands")
            return positions
        except Exception as e:
            logger.error(f"Failed to infer market positions: {e}")
            return {}

    def get_inferred_facts(self) -> list[dict[str, Any]]:
        """
        추론된 사실들 조회

        Returns:
            추론 결과 리스트
        """
        if not OWLREADY2_AVAILABLE:
            return []

        facts = []

        try:
            # Brand 서브클래스 추론 결과
            for brand in self.onto.Brand.instances():
                if self.onto.DominantBrand in brand.is_a:
                    facts.append(
                        {
                            "type": "market_position",
                            "subject": brand.name,
                            "position": "DominantBrand",
                            "sos": brand.share_of_shelf or 0.0,
                        }
                    )
                elif self.onto.StrongBrand in brand.is_a:
                    facts.append(
                        {
                            "type": "market_position",
                            "subject": brand.name,
                            "position": "StrongBrand",
                            "sos": brand.share_of_shelf or 0.0,
                        }
                    )
                elif self.onto.NicheBrand in brand.is_a:
                    facts.append(
                        {
                            "type": "market_position",
                            "subject": brand.name,
                            "position": "NicheBrand",
                            "sos": brand.share_of_shelf or 0.0,
                        }
                    )

            # 대칭 관계 추론 (competsWith)
            for brand in self.onto.Brand.instances():
                for competitor in brand.competes_with:
                    facts.append(
                        {
                            "type": "competition",
                            "subject": brand.name,
                            "object": competitor.name,
                            "relation": "competsWith",
                        }
                    )

            logger.info(f"Retrieved {len(facts)} inferred facts")
            return facts
        except Exception as e:
            logger.error(f"Failed to get inferred facts: {e}")
            return []

    # =========================================================================
    # 쿼리 기능
    # =========================================================================

    def query_sparql(self, sparql_query: str) -> list[Any]:
        """
        SPARQL 쿼리 실행

        Args:
            sparql_query: SPARQL 쿼리 문자열

        Returns:
            쿼리 결과 리스트
        """
        if not OWLREADY2_AVAILABLE:
            logger.warning("SPARQL not available without owlready2")
            return []

        try:
            # owlready2의 search() 메서드로 간단한 쿼리 구현
            # 전체 SPARQL 지원은 rdflib 필요
            logger.warning("Full SPARQL not implemented. Use query methods instead.")
            return []
        except Exception as e:
            logger.error(f"SPARQL query failed: {e}")
            return []

    def get_brand_info(self, brand_name: str) -> dict[str, Any] | None:
        """
        브랜드 정보 조회

        Args:
            brand_name: 브랜드명

        Returns:
            브랜드 정보
        """
        if not OWLREADY2_AVAILABLE:
            return None

        try:
            brand_id = brand_name.replace(" ", "_")
            brand = self.onto.search_one(iri=f"*{brand_id}")

            if not brand:
                return None

            return {
                "name": brand.name,
                "sos": brand.share_of_shelf or 0.0,
                "avg_rank": brand.average_rank,
                "product_count": brand.product_count or 0,
                "products": [p.name for p in brand.has_product] if brand.has_product else [],
                "competitors": [c.name for c in brand.competes_with] if brand.competes_with else [],
                "market_position": self._get_brand_position(brand),
            }
        except Exception as e:
            logger.error(f"Failed to get brand info for {brand_name}: {e}")
            return None

    def get_competitors(self, brand_name: str) -> list[str]:
        """
        경쟁사 조회 (대칭 관계 활용)

        Args:
            brand_name: 브랜드명

        Returns:
            경쟁사 리스트
        """
        if not OWLREADY2_AVAILABLE:
            return []

        try:
            brand_id = brand_name.replace(" ", "_")
            brand = self.onto.search_one(iri=f"*{brand_id}")

            if not brand:
                return []

            return [c.name for c in brand.competes_with]
        except Exception as e:
            logger.error(f"Failed to get competitors for {brand_name}: {e}")
            return []

    def get_all_brands(self) -> list[dict[str, Any]]:
        """
        모든 브랜드 정보 조회

        Returns:
            브랜드 정보 리스트
        """
        if not OWLREADY2_AVAILABLE:
            return []

        brands = []
        try:
            for brand in self.onto.Brand.instances():
                brands.append(
                    {
                        "name": brand.name,
                        "share_of_shelf": brand.share_of_shelf or 0.0,
                        "average_rank": brand.average_rank,
                        "product_count": brand.product_count or 0,
                        "market_position": self._get_brand_position(brand),
                        "competitors": [c.name for c in brand.competes_with]
                        if brand.competes_with
                        else [],
                    }
                )
            return brands
        except Exception as e:
            logger.error(f"Failed to get all brands: {e}")
            return []

    def get_category_brands(self, category_id: str) -> list[dict[str, Any]]:
        """
        카테고리별 브랜드 조회

        Args:
            category_id: 카테고리 ID

        Returns:
            브랜드 정보 리스트
        """
        if not OWLREADY2_AVAILABLE:
            return []

        try:
            category = self.onto.search_one(iri=f"*{category_id}")
            if not category:
                return []

            # 카테고리에 속한 제품들
            products = self.onto.search(belongs_to_category=category)

            # 제품별 브랜드 추출
            brands_dict = {}
            for product in products:
                if product.has_brand:
                    brand = product.has_brand[0]
                    brand_name = brand.name
                    if brand_name not in brands_dict:
                        brands_dict[brand_name] = {
                            "brand": brand_name,
                            "sos": brand.share_of_shelf or 0.0,
                            "product_count": 0,
                        }
                    brands_dict[brand_name]["product_count"] += 1

            return sorted(brands_dict.values(), key=lambda x: x["sos"], reverse=True)
        except Exception as e:
            logger.error(f"Failed to get brands for category {category_id}: {e}")
            return []

    def get_brand_market_position(self, brand_name: str) -> str | None:
        """
        브랜드의 시장 포지션 조회

        Args:
            brand_name: 브랜드명

        Returns:
            "DominantBrand", "StrongBrand", "NicheBrand" 중 하나 (또는 None)
        """
        if not OWLREADY2_AVAILABLE:
            return None

        try:
            brand_id = brand_name.replace(" ", "_")
            brand = self.onto.search_one(iri=f"*{brand_id}")

            if not brand:
                return None

            return self._get_brand_position(brand)
        except Exception as e:
            logger.error(f"Failed to get market position for {brand_name}: {e}")
            return None

    def _get_brand_position(self, brand) -> str | None:
        """브랜드 인스턴스에서 포지션 추출"""
        if self.onto.DominantBrand in brand.is_a:
            return "DominantBrand"
        elif self.onto.StrongBrand in brand.is_a:
            return "StrongBrand"
        elif self.onto.NicheBrand in brand.is_a:
            return "NicheBrand"
        return None

    # =========================================================================
    # 데이터 마이그레이션
    # =========================================================================

    def import_from_knowledge_graph(self, kg: Any) -> int:
        """
        기존 KnowledgeGraph 데이터를 OWL로 변환

        Args:
            kg: KnowledgeGraph 인스턴스

        Returns:
            변환된 엔티티 수
        """
        if not OWLREADY2_AVAILABLE:
            logger.warning("Cannot import without owlready2")
            return 0

        count = 0

        try:
            # 브랜드 메타데이터 → OWL Brand
            for entity, metadata in kg.entity_metadata.items():
                if metadata.get("type") == "brand":
                    self.add_brand(
                        name=entity,
                        sos=metadata.get("sos", 0.0),
                        avg_rank=metadata.get("avg_rank"),
                        product_count=metadata.get("product_count", 0),
                    )
                    count += 1

            # 제품 관계 → OWL Product
            from .relations import RelationType

            for rel in kg.query(predicate=RelationType.HAS_PRODUCT):
                brand = rel.subject
                asin = rel.object
                props = rel.properties

                self.add_product(
                    asin=asin,
                    brand=brand,
                    category=props.get("category", "unknown"),
                    rank=props.get("rank", 999),
                    price=props.get("price"),
                    rating=props.get("rating"),
                )
                count += 1

            # 경쟁 관계 → OWL competsWith
            for rel in kg.query(predicate=RelationType.COMPETES_WITH):
                self.add_competitor_relation(rel.subject, rel.object)

            logger.info(f"Imported {count} entities from KnowledgeGraph")
            return count
        except Exception as e:
            logger.error(f"Failed to import from KnowledgeGraph: {e}")
            return count

    def import_from_metrics(self, metrics_data: dict[str, Any]) -> int:
        """
        메트릭 데이터를 OWL로 변환

        Args:
            metrics_data: MetricsAgent 결과

        Returns:
            변환된 브랜드 수
        """
        if not OWLREADY2_AVAILABLE:
            return 0

        count = 0

        try:
            for brand_metric in metrics_data.get("brand_metrics", []):
                brand = brand_metric.get("brand_name")
                if brand:
                    self.add_brand(
                        name=brand,
                        sos=brand_metric.get("share_of_shelf", 0.0),
                        avg_rank=brand_metric.get("avg_rank"),
                        product_count=brand_metric.get("product_count", 0),
                    )
                    count += 1

            logger.info(f"Imported {count} brands from metrics data")
            return count
        except Exception as e:
            logger.error(f"Failed to import from metrics: {e}")
            return count

    # =========================================================================
    # 영속화
    # =========================================================================

    def save(self, path: str | None = None) -> bool:
        """
        OWL 파일 저장

        Args:
            path: 저장 경로 (None이면 초기화 시 설정된 경로)

        Returns:
            성공 여부
        """
        if not OWLREADY2_AVAILABLE or not self.onto:
            return False

        try:
            save_path = Path(path) if path else self.owl_file
            if not save_path:
                save_path = Path("data/ontology/amore_brand.owl")

            save_path.parent.mkdir(parents=True, exist_ok=True)

            self.onto.save(file=str(save_path), format="rdfxml")
            logger.info(f"Saved ontology to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save ontology: {e}")
            return False

    def load(self, path: str) -> bool:
        """
        OWL 파일 로드

        Args:
            path: OWL 파일 경로

        Returns:
            성공 여부
        """
        if not OWLREADY2_AVAILABLE:
            return False

        try:
            self.onto = get_ontology(f"file://{Path(path).absolute()}").load()
            logger.info(f"Loaded ontology from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load ontology from {path}: {e}")
            return False

    # =========================================================================
    # 유틸리티
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """온톨로지 통계"""
        if not OWLREADY2_AVAILABLE or not self.onto:
            return {"error": "owlready2 not available"}

        try:
            return {
                "brands": len(list(self.onto.Brand.instances())),
                "products": len(list(self.onto.Product.instances())),
                "categories": len(list(self.onto.Category.instances())),
                "trends": len(list(self.onto.Trend.instances())),
                "dominant_brands": len(list(self.onto.DominantBrand.instances())),
                "strong_brands": len(list(self.onto.StrongBrand.instances())),
                "niche_brands": len(list(self.onto.NicheBrand.instances())),
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    def clear(self):
        """온톨로지 초기화 (모든 인스턴스 삭제)"""
        if not OWLREADY2_AVAILABLE or not self.onto:
            return

        try:
            with self.onto:
                # 모든 인스턴스 삭제
                for individual in list(self.onto.individuals()):
                    destroy_entity(individual)

            logger.info("Cleared all individuals from ontology")
        except Exception as e:
            logger.error(f"Failed to clear ontology: {e}")

    def __repr__(self):
        stats = self.get_stats()
        return f"OWLReasoner(brands={stats.get('brands', 0)}, products={stats.get('products', 0)})"


# =========================================================================
# 싱글톤 패턴
# =========================================================================

_owl_reasoner_instance: OWLReasoner | None = None


def get_owl_reasoner(owl_file: str | None = None, reasoner_type: str = "pellet") -> OWLReasoner:
    """
    OWLReasoner 싱글톤 인스턴스 반환

    Args:
        owl_file: OWL 파일 경로
        reasoner_type: "pellet" 또는 "hermit"

    Returns:
        OWLReasoner 인스턴스
    """
    global _owl_reasoner_instance
    if _owl_reasoner_instance is None:
        _owl_reasoner_instance = OWLReasoner(owl_file=owl_file, reasoner_type=reasoner_type)
    return _owl_reasoner_instance
