# prompts/ - LLM 프롬프트 템플릿

## 개요

AI 에이전트가 사용하는 LLM 프롬프트 템플릿을 중앙 관리합니다.
시스템 프롬프트, 쿼리 라우팅, 인사이트 생성 등 역할별로 분리되어 있습니다.

## 파일 목록

| 파일 | LOC | 설명 |
|------|-----|------|
| `chat_system.txt` | ~150 | 챗봇 시스템 프롬프트 |
| `insight_generation.txt` | ~200 | 인사이트 생성 프롬프트 |
| `query_router.txt` | ~100 | 쿼리 분류 프롬프트 |
| `metrics.json` | ~50 | 지표 설명 및 해석 |
| `version_manager.py` | ~150 | 프롬프트 버전 관리 |
| `components/` | - | 재사용 가능한 프롬프트 컴포넌트 |
| `AGENTS.md` | - | 현재 문서 |

## chat_system.txt

### 역할

AI 챗봇의 시스템 프롬프트입니다.
역할, 능력, 제약사항, 응답 스타일을 정의합니다.

### 구조

```
# 역할 정의
당신은 AMORE Pacific의 Amazon 시장 분석 AI 어시스턴트입니다.
LANEIGE 브랜드의 경쟁력을 모니터링하고 전략적 인사이트를 제공합니다.

# 지식 범위
- Amazon Top 100 제품 데이터 (5개 카테고리)
- 시장 점유율 (SoS), 경쟁 지수 (HHI), 가격 경쟁력 (CPI)
- LANEIGE vs 경쟁사 (Summer Fridays, Aquaphor, Burt's Bees 등)

# 응답 스타일
- 간결하고 데이터 기반
- 구체적인 수치 제시
- 전략적 시사점 포함

# 제약사항
- 추측 금지 (데이터 없으면 "정보 없음" 명시)
- 거짓 정보 생성 금지 (Hallucination 방지)
- 개인정보 및 민감 정보 보호
```

### 사용 예시

```python
# src/agents/hybrid_chatbot_agent.py
system_prompt = open("prompts/chat_system.txt").read()

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "LANEIGE 1위 제품은?"}
]

response = await llm_client.chat(messages)
```

### 버전 히스토리

| 버전 | 날짜 | 변경사항 |
|------|------|----------|
| v1.0 | 2026-01-15 | 초기 버전 |
| v1.1 | 2026-01-20 | 응답 스타일 개선 (간결성) |
| v1.2 | 2026-02-01 | Hallucination 방지 강화 |

## insight_generation.txt

### 역할

전략적 인사이트 생성 프롬프트입니다.
데이터 분석 결과를 경영진 대상 보고서로 변환합니다.

### 구조

```
# 목표
다음 데이터를 기반으로 LANEIGE의 시장 포지셔닝과 전략적 시사점을 도출하세요.

# 입력 데이터
- 카테고리별 순위: {rankings}
- 시장 점유율: SoS {sos}%, HHI {hhi}, CPI {cpi}
- 순위 변동: {rank_changes}
- 경쟁사 동향: {competitor_trends}
- 외부 신호: {external_signals}

# 출력 형식
## 요약 (2-3문장)
핵심 발견사항을 간결하게 요약합니다.

## 주요 발견
1. 순위 변동 및 원인
2. 시장 점유율 변화
3. 경쟁 환경 분석

## 전략적 시사점
1. 기회 (Opportunities)
2. 위험 (Threats)
3. 권장 액션 (Recommendations)

# 제약사항
- 데이터에 기반한 사실만 기술
- 추측성 표현 지양
- 구체적인 수치 및 근거 제시
```

### 사용 예시

```python
# src/agents/hybrid_insight_agent.py
prompt_template = open("prompts/insight_generation.txt").read()

prompt = prompt_template.format(
    rankings=rankings_data,
    sos=45.2,
    hhi=0.15,
    cpi=0.89,
    rank_changes=changes,
    competitor_trends=trends,
    external_signals=signals
)

insight = await llm_client.chat([
    {"role": "system", "content": "You are a strategic analyst."},
    {"role": "user", "content": prompt}
])
```

### 예시 출력

```
## 요약
LANEIGE Lip Sleeping Mask는 Lip Care 카테고리에서 1위를 유지하고 있으며,
시장 점유율은 45.2%로 Summer Fridays 대비 2배 높습니다.

## 주요 발견
1. 순위 변동: 지난주 대비 순위 변동 없음 (1위 유지)
2. 시장 점유율: SoS 45.2% (전월 대비 +2.3%p 증가)
3. 경쟁 환경: HHI 0.15로 경쟁이 치열한 시장

## 전략적 시사점
1. 기회: TikTok에서 "Lip Sleeping Mask" 검색량 급증 (전월 대비 +35%)
2. 위험: Summer Fridays가 신제품 출시로 공격적 마케팅 중
3. 권장 액션: 소셜 미디어 마케팅 강화 및 리뷰 관리 필요
```

## query_router.txt

### 역할

사용자 쿼리를 유형별로 분류하는 프롬프트입니다.
RAG, KG, Ontology 중 적절한 검색 방법을 선택합니다.

### 구조

```
# 목표
다음 사용자 쿼리를 분석하여 적절한 검색 방법을 선택하세요.

# 쿼리 유형
1. Factual: 사실 확인 (예: "1위 제품은?")
   → 검색 방법: KG (Knowledge Graph)

2. Analytical: 분석 및 계산 (예: "시장 점유율은?")
   → 검색 방법: RAG + KG + Ontology

3. Comparative: 비교 분석 (예: "A vs B 비교해줘")
   → 검색 방법: RAG + KG

4. Temporal: 시계열 분석 (예: "지난달 대비 변동은?")
   → 검색 방법: KG + Ontology

# 출력 형식
{
  "query_type": "factual | analytical | comparative | temporal",
  "search_method": ["rag", "kg", "ontology"],
  "complexity": "simple | medium | complex",
  "requires_calculation": true | false
}

# 사용자 쿼리
{query}
```

### 사용 예시

```python
# src/rag/router.py
prompt_template = open("prompts/query_router.txt").read()
prompt = prompt_template.format(query="LANEIGE 1위 제품은?")

routing_result = await llm_client.chat([
    {"role": "user", "content": prompt}
])

# {'query_type': 'factual', 'search_method': ['kg'], 'complexity': 'simple'}
```

## metrics.json

### 역할

지표 정의 및 해석 가이드입니다.
SoS, HHI, CPI의 계산식과 의미를 설명합니다.

### 구조

```json
{
  "sos": {
    "name": "Share of Shelf",
    "description": "시장 점유율",
    "formula": "(브랜드 제품 수 / 전체 제품 수) × 100",
    "interpretation": {
      "excellent": "> 50%",
      "good": "30-50%",
      "fair": "10-30%",
      "poor": "< 10%"
    },
    "example": "SoS 45.2%는 Top 100 중 45개가 LANEIGE 제품임을 의미"
  },
  "hhi": {
    "name": "Herfindahl-Hirschman Index",
    "description": "시장 집중도",
    "formula": "Σ(각 브랜드 점유율)^2",
    "interpretation": {
      "concentrated": "> 0.25",
      "moderate": "0.15-0.25",
      "competitive": "< 0.15"
    },
    "example": "HHI 0.15는 경쟁이 치열한 시장을 의미"
  },
  "cpi": {
    "name": "Competitive Price Index",
    "description": "가격 경쟁력",
    "formula": "브랜드 평균 가격 / 시장 평균 가격",
    "interpretation": {
      "premium": "> 1.2",
      "competitive": "0.8-1.2",
      "value": "< 0.8"
    },
    "example": "CPI 0.89는 시장 평균보다 11% 저렴함을 의미"
  }
}
```

### 사용 예시

```python
import json

metrics_guide = json.load(open("prompts/metrics.json"))
sos_desc = metrics_guide["sos"]["description"]
# "시장 점유율"
```

## version_manager.py

### 역할

프롬프트 버전 관리 및 A/B 테스트를 지원합니다.

### 주요 기능

```python
from prompts.version_manager import PromptVersionManager

manager = PromptVersionManager()

# 버전 등록
manager.register_version(
    name="chat_system",
    version="v1.2",
    path="prompts/chat_system.txt"
)

# 프롬프트 로드
prompt = manager.load("chat_system", version="v1.2")

# A/B 테스트
manager.set_ab_test(
    name="chat_system",
    version_a="v1.1",
    version_b="v1.2",
    split_ratio=0.5
)

# 랜덤 선택
prompt = manager.load_with_ab_test("chat_system")
```

### 버전 히스토리 저장

```json
// prompts/.versions/chat_system.json
{
  "versions": [
    {
      "version": "v1.0",
      "created_at": "2026-01-15T10:00:00",
      "author": "admin",
      "changes": "초기 버전"
    },
    {
      "version": "v1.1",
      "created_at": "2026-01-20T14:30:00",
      "author": "admin",
      "changes": "응답 스타일 개선"
    }
  ]
}
```

## components/

재사용 가능한 프롬프트 컴포넌트입니다.

### 구조

```
components/
├── role_definitions.txt       # 역할 정의
├── constraints.txt            # 제약사항
├── output_formats.txt         # 출력 형식
└── examples.txt               # 예시
```

### 사용 예시

```python
# 컴포넌트 조합
role = open("prompts/components/role_definitions.txt").read()
constraints = open("prompts/components/constraints.txt").read()

system_prompt = f"{role}\n\n{constraints}"
```

## 프롬프트 엔지니어링 가이드

### 1. 명확한 역할 정의

```
❌ "당신은 AI입니다."
✅ "당신은 AMORE Pacific의 Amazon 시장 분석 전문가입니다."
```

### 2. 구체적인 제약사항

```
❌ "정확하게 답변하세요."
✅ "데이터가 없으면 '정보 없음'이라고 명시하고, 추측하지 마세요."
```

### 3. 출력 형식 지정

```
❌ "분석해주세요."
✅ "다음 형식으로 분석해주세요: 1. 요약 2. 주요 발견 3. 시사점"
```

### 4. Few-shot Examples

```
# 좋은 예시
Query: "LANEIGE 1위 제품은?"
Answer: "LANEIGE Lip Sleeping Mask (ASIN: B0CDJQTY77)가 Lip Care 카테고리에서 1위입니다."

# 나쁜 예시
Query: "LANEIGE 1위 제품은?"
Answer: "아마도 립밤일 것 같습니다."  # 추측 금지!
```

## 주의사항

1. **Hallucination 방지**: 데이터 없으면 명시적으로 "정보 없음"
2. **버전 관리**: 프롬프트 변경 시 버전 번호 증가
3. **A/B 테스트**: 새 프롬프트는 A/B 테스트로 검증
4. **토큰 효율**: 불필요한 설명 제거 (비용 절감)
5. **다국어 지원**: 한국어/영어 버전 분리 관리

## 평가 메트릭

| 메트릭 | 설명 | 목표 |
|--------|------|------|
| Relevance | 질문 관련성 | ≥ 0.9 |
| Correctness | 사실적 정확성 | ≥ 0.85 |
| Completeness | 답변 완성도 | ≥ 0.8 |
| Conciseness | 간결성 | ≤ 200 tokens |

## 개선 방향

- [ ] 다국어 프롬프트 (영어, 한국어)
- [ ] 프롬프트 자동 최적화 (DSPy)
- [ ] 실시간 A/B 테스트 대시보드
- [ ] 프롬프트 성능 모니터링
- [ ] 사용자 피드백 기반 개선

## 참고

- `src/agents/hybrid_chatbot_agent.py` - 챗봇 프롬프트 사용
- `src/agents/hybrid_insight_agent.py` - 인사이트 프롬프트 사용
- `src/rag/router.py` - 쿼리 라우팅 프롬프트 사용
- `eval/` - 프롬프트 평가 프레임워크
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
