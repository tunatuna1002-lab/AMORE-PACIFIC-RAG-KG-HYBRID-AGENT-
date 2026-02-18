# src/shared/ - Cross-Cutting Concerns

## 개요

모든 레이어에서 공통으로 사용되는 유틸리티와 공유 컴포넌트를 제공합니다.
Clean Architecture의 어느 레이어에도 속하지 않는 횡단 관심사(Cross-Cutting Concerns)를 다룹니다.

## 역할

- 전역 상수 정의
- LLM 클라이언트 래퍼
- 공통 유틸리티 함수
- 로깅, 설정 관리

## 파일 목록

| 파일 | LOC | 설명 |
|------|-----|------|
| `constants.py` | ~50 | 전역 상수 정의 |
| `llm_client.py` | ~200 | LiteLLM 래퍼 (GPT-4.1-mini) |
| `__init__.py` | 1 | 패키지 초기화 |
| `AGENTS.md` | - | 현재 문서 |

## constants.py

### 정의된 상수

```python
# 카테고리 상수
CATEGORIES = ["beauty", "lip_care", "lip_makeup", "powder", "skin_care"]

# 브랜드 상수
TARGET_BRAND = "LANEIGE"
TRACKED_BRANDS = ["LANEIGE", "Summer Fridays", "Laneige", ...]

# 파일 경로
DATA_DIR = "data"
KG_PATH = "data/knowledge_graph.json"
DB_PATH = "data/amore_data.db"

# API 설정
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
```

### 사용 예시

```python
from src.shared.constants import TARGET_BRAND, CATEGORIES

for category in CATEGORIES:
    products = await crawler.crawl(category, brand=TARGET_BRAND)
```

## llm_client.py

### LLMClient 클래스

LiteLLM을 래핑하여 OpenAI API 호출을 추상화합니다.

```python
from src.shared.llm_client import LLMClient

client = LLMClient(
    model="gpt-4.1-mini",
    temperature=0.4,
    max_tokens=1500
)

response = await client.chat(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Analyze this data..."}
    ]
)
```

### 주요 메서드

| 메서드 | 설명 |
|--------|------|
| `chat(messages, **kwargs)` | 채팅 완성 (async) |
| `embed(text)` | 텍스트 임베딩 생성 |
| `count_tokens(text)` | 토큰 수 계산 |

### 환경 변수

```bash
# .env
OPENAI_API_KEY=sk-...
LLM_TEMPERATURE_CHAT=0.4       # 챗봇 temperature
LLM_TEMPERATURE_INSIGHT=0.6    # 인사이트 temperature
```

### 모델 설정

| 용도 | 모델 | Temperature |
|------|------|-------------|
| 챗봇 | `gpt-4.1-mini` | 0.4 |
| 인사이트 | `gpt-4.1-mini` | 0.6 |
| 임베딩 | `text-embedding-3-small` | - |

### Error Handling

```python
try:
    response = await client.chat(messages)
except Exception as e:
    logger.error(f"LLM API 호출 실패: {e}")
    # Graceful degradation
```

## 의존성

### External Dependencies

```python
from litellm import acompletion  # LiteLLM async completion
import tiktoken  # OpenAI tokenizer
```

### 허용되는 Import

- 모든 레이어에서 `src.shared.*` import 가능
- shared는 다른 레이어를 import하지 않음 (순환 참조 방지)

```python
# ✅ 모든 레이어에서 사용 가능
from src.shared.constants import TARGET_BRAND
from src.shared.llm_client import LLMClient

# ❌ shared가 다른 레이어 참조 금지
from src.domain.entities.product import Product  # ❌
```

## 주의사항

1. **순환 참조 금지**: shared는 domain/application/infrastructure 참조 불가
2. **순수 유틸리티**: 비즈니스 로직 포함 금지
3. **상태 없음**: 가능한 stateless 함수/클래스 유지
4. **환경 변수**: 민감 정보는 `.env`에서 관리
5. **API Key 보안**: 로그에 API Key 노출 방지 (마스킹 처리)

## LLM 비용 최적화

### 캐싱 전략

```python
# Embedding 캐시 (src/rag/retriever.py)
embedding_cache = {}  # SHA-256 해시 → 벡터 캐싱
```

### 토큰 절약 팁

- 시스템 프롬프트 간결화
- 불필요한 컨텍스트 제거
- `max_tokens` 제한 설정
- Streaming 응답 활용

## 테스트

```bash
# Shared 모듈 테스트
python3 -m pytest tests/unit/shared/ -v
```

## 참고

- [LiteLLM 문서](https://docs.litellm.ai/)
- [OpenAI Tokenizer](https://platform.openai.com/tokenizer)
- `src/rag/retriever.py` - Embedding 캐시 구현
- `prompts/` - LLM 프롬프트 템플릿
