# AMORE RAG-KG 하이브리드 에이전트 구현 계획서

> 작성일: 2025-01-19
> 버전: v1.0
> 상태: Draft

---

## 1. 프로젝트 개요

### 1.1 목적
아모레퍼시픽 멘토링 피드백을 반영하여 LANEIGE 브랜드 경쟁력 분석 시스템 고도화

### 1.2 핵심 요구사항 (회의록 기반)

| # | 요구사항 | 우선순위 | 현재 상태 |
|---|---------|---------|----------|
| 1 | 가격/할인 분석 | 🔴 High | 60% |
| 2 | 경쟁사 정보 (Summer Fridays 등) | 🔴 High | 46% |
| 3 | 카테고리 계층 구조 | 🔴 High | 30% |
| 4 | 인과관계 규명 (할인↔순위) | 🟡 Medium | 34% |
| 5 | AI 출처 표시 | 🟢 Low | 85% |
| 6 | 시점 정보 추가 | 🟢 Low | 73% |
| 7 | 글로벌 확장성 | ⚪ 보류 | - |

---

## 2. 아키텍처 변경 사항

### 2.1 카테고리 계층 구조 (신규)

```
Beauty & Personal Care (beauty)
├── Skin Care (11060451)
│   ├── Body (11060521)
│   ├── Eyes (11061941)
│   ├── Face (11060711)
│   ├── Lip Care (3761351)  ← 모니터링 대상
│   ├── Maternity (11062371)
│   ├── Sets & Kits (11062581)
│   └── Sunscreens & Tanning (11062591)
│
├── Makeup (11058281)
│   ├── Eyes
│   ├── Face
│   ├── Lips (lip_makeup)  ← 모니터링 대상
│   ├── Makeup Palettes
│   └── Makeup Remover
│
└── (기타 카테고리...)
```

### 2.2 데이터 모델 변경

#### schema.py 수정
```python
class RankRecord(BaseModel):
    # 기존 필드...

    # 신규 필드
    collected_at: Optional[datetime] = None  # 정확한 수집 시간
    discount_trend: Optional[str] = None     # "up", "down", "stable"
    previous_price: Optional[float] = None   # 이전 가격 (할인 추적용)
    previous_discount: Optional[float] = None # 이전 할인율

class Category(BaseModel):
    category_id: str
    name: str
    amazon_node_id: str
    level: int = 0                           # 계층 레벨
    parent_id: Optional[str] = None          # 부모 카테고리
    path: List[str] = []                     # 전체 경로
    url: Optional[str] = None
```

### 2.3 Knowledge Graph 확장

#### 신규 관계 타입
```python
class RelationType(str, Enum):
    # 기존...

    # 신규: 할인 인과관계
    DISCOUNT_CORRELATES_RANK = "discountCorrelatesRank"
    PRICE_AFFECTS_RANK = "priceAffectsRank"

    # 신규: 경쟁사 가격 비교
    PRICED_HIGHER_THAN = "pricedHigherThan"
    PRICED_LOWER_THAN = "pricedLowerThan"
```

---

## 3. 구현 Phase

### Phase 1: 기반 작업 (Day 1-2)

#### Task 1.1: 스키마 업데이트
- [ ] `schema.py`: RankRecord에 `collected_at`, `discount_trend` 필드 추가
- [ ] `schema.py`: Category 모델에 계층 필드 추가
- [ ] 테스트: 기존 데이터 호환성 확인

#### Task 1.2: 카테고리 계층 구조 구현
- [ ] `config/category_hierarchy.json` 생성 (아마존에서 수집한 데이터)
- [ ] `knowledge_graph.py`: `load_category_hierarchy()` 메서드 추가
- [ ] `relations.py`: 필요시 관계 타입 추가

#### Task 1.3: 크롤러 개선
- [ ] `amazon_scraper.py`: `collected_at` 타임스탬프 추가
- [ ] `amazon_scraper.py`: 할인 변화 추적 로직

### Phase 2: 핵심 기능 (Day 3-5)

#### Task 2.1: 대시보드 할인 정보 표시
- [ ] 할인율 카드/배지 추가
- [ ] 할인 추이 그래프 (있다면 개선)
- [ ] 쿠폰 정보 표시

#### Task 2.2: 경쟁사 분석 시스템
- [ ] `config/competitors.json` 생성
  - 고정 경쟁사: Summer Fridays, COSRX 등
  - 유동 경쟁사: AI 감지 로직
- [ ] 대시보드: 경쟁사 선택 토글/검색 UI
- [ ] 챗봇: 경쟁사 비교 쿼리 지원

#### Task 2.3: 챗봇 카테고리 인식 개선
- [ ] `hybrid_chatbot_agent.py`: 카테고리 계층 컨텍스트 추가
- [ ] "Lip Care 4위 vs Beauty 전체 73위" 설명 가능하도록

### Phase 3: 고도화 (Day 6-8)

#### Task 3.1: 순위-할인 인과관계 분석
- [ ] 비즈니스 규칙 추가: 할인과 순위 상관관계
- [ ] 가격 효과 태그: 할인 의존형(빨강), 바이럴 효과(녹색)
- [ ] 할인 의존도 지표 계산

#### Task 3.2: Today's Deals 크롤링 (선택)
- [ ] 별도 크롤러 또는 기존 확장
- [ ] ASIN 기준 데이터 조인

#### Task 3.3: 출처 표시 강화
- [ ] 구체적 URL, 페이지 참조 추가
- [ ] Perplexity/Liner 스타일 상세 출처

### Phase 4: 마무리 (Day 9-10)

#### Task 4.1: 테스트 및 검증
- [ ] 단위 테스트
- [ ] 통합 테스트
- [ ] 대시보드 UI/UX 검토

#### Task 4.2: 문서화
- [ ] 데이터 정의서 (지표 설명)
- [ ] API 문서 업데이트
- [ ] 발표 자료 지원

---

## 4. 파일별 변경 사항

### 4.1 신규 파일

| 파일 | 설명 |
|------|------|
| `config/category_hierarchy.json` | 아마존 카테고리 계층 구조 |
| `config/competitors.json` | 고정/유동 경쟁사 목록 |
| `docs/guides/Data_Definition.md` | 데이터 정의서 |

### 4.2 수정 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/ontology/schema.py` | RankRecord, Category 필드 추가 |
| `src/ontology/relations.py` | 할인 인과관계 타입 추가 |
| `src/ontology/knowledge_graph.py` | 카테고리 계층 로딩 메서드 |
| `src/tools/amazon_scraper.py` | collected_at, 할인 추적 |
| `src/agents/hybrid_chatbot_agent.py` | 카테고리 컨텍스트, 출처 강화 |
| `dashboard/amore_unified_dashboard_v4.html` | 할인 섹션, 경쟁사 UI |

---

## 5. 카테고리 계층 데이터 (수집 완료)

```json
{
  "beauty": {
    "name": "Beauty & Personal Care",
    "amazon_node_id": "beauty",
    "level": 0,
    "parent_id": null,
    "children": ["skin_care", "makeup", "hair_care", "fragrance", "personal_care"]
  },
  "skin_care": {
    "name": "Skin Care",
    "amazon_node_id": "11060451",
    "level": 1,
    "parent_id": "beauty",
    "children": ["body_skincare", "eyes_skincare", "face_skincare", "lip_care", "maternity", "sets_kits", "sunscreens"]
  },
  "lip_care": {
    "name": "Lip Care",
    "amazon_node_id": "3761351",
    "level": 2,
    "parent_id": "skin_care",
    "path": ["beauty", "skin_care", "lip_care"],
    "children": []
  },
  "makeup": {
    "name": "Makeup",
    "amazon_node_id": "11058281",
    "level": 1,
    "parent_id": "beauty",
    "children": ["eyes_makeup", "face_makeup", "lips_makeup", "makeup_palettes", "makeup_remover", "makeup_sets"]
  },
  "lip_makeup": {
    "name": "Lips",
    "amazon_node_id": "lips_makeup_node",
    "level": 2,
    "parent_id": "makeup",
    "path": ["beauty", "makeup", "lip_makeup"],
    "children": []
  },
  "face_powder": {
    "name": "Face Powder",
    "amazon_node_id": "face_powder_node",
    "level": 2,
    "parent_id": "makeup",
    "path": ["beauty", "makeup", "face_powder"],
    "children": []
  }
}
```

---

## 6. 리스크 및 대응

| 리스크 | 영향 | 대응 방안 |
|--------|------|----------|
| 아마존 크롤링 차단 | 🔴 High | User-Agent 로테이션, 요청 간격 조절 |
| 할인 데이터 정확도 | 🟡 Medium | 여러 시점 데이터 교차 검증 |
| 카테고리 ID 변경 | 🟡 Medium | 동적 매핑 테이블 유지 |
| 공모전 일정 압박 | 🟡 Medium | Phase 1-2 우선, Phase 3 선택적 |

---

## 7. 성공 지표

### 7.1 기능 완성도
- [ ] 할인 정보가 대시보드에 표시됨
- [ ] 경쟁사 선택 및 비교 가능
- [ ] 챗봇이 카테고리 계층 인식하여 응답
- [ ] AI 분석 결과에 출처 명시

### 7.2 사용자 시나리오
- [ ] "LANEIGE Lip Care 4위가 전체 Beauty에서는 몇 위인가요?" → 정확한 답변
- [ ] "Summer Fridays와 가격 비교해줘" → 경쟁사 비교 제공
- [ ] "최근 할인 때문에 순위가 오른 건가요?" → 인과관계 분석

---

## 8. 다음 단계

1. **계획 검토**: 이 계획서를 팀과 공유하여 피드백
2. **우선순위 확정**: Phase 1-2 필수, Phase 3 선택
3. **구현 시작**: `/sisyphus` 모드로 병렬 작업 진행

---

> 이 계획서는 Prometheus 플래닝 세션에서 생성되었습니다.
> 검토: `/review` 명령으로 Momus 비평 가능
