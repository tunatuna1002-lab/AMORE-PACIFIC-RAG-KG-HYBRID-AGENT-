# TDD 기반 리팩토링 계획서

> 생성일: 2026-01-23
> 목표: 분석에서 발견된 이슈들을 TDD 방식으로 해결

---

## 📋 요구사항 요약

| 항목 | 결정 |
|------|------|
| 테스트 프레임워크 | pytest (기존 pytest.ini 활용) |
| 진행 순서 | 예외 처리 → 테스트 → 입력검증 → DI |
| DI 범위 | 핵심 에이전트 3개 (Insight, Chatbot, Crawler) |
| 커버리지 목표 | 70% |

---

## 🎯 Phase 1: 예외 처리 개선 (TDD)

### 목표
광범위한 `except Exception` 대신 구체적인 예외 타입 정의

### 테스트 먼저 (RED)
```python
# tests/unit/domain/test_exceptions.py
def test_network_error_attributes():
    """NetworkError는 url, status_code, retry_count 속성 가져야 함"""

def test_llm_api_error_attributes():
    """LLMAPIError는 model, error_code, is_retryable 속성 가져야 함"""

def test_validation_error_attributes():
    """DataValidationError는 field, value, constraint 속성 가져야 함"""

def test_scraper_error_attributes():
    """ScraperError는 category, asin, error_type 속성 가져야 함"""
```

### 구현 (GREEN)
```python
# src/domain/exceptions.py

class AmoreAgentError(Exception):
    """Base exception for all AMORE agent errors"""
    pass

class NetworkError(AmoreAgentError):
    """Network-related errors (timeout, connection)"""
    def __init__(self, message: str, url: str = None,
                 status_code: int = None, retry_count: int = 0):
        super().__init__(message)
        self.url = url
        self.status_code = status_code
        self.retry_count = retry_count

class LLMAPIError(AmoreAgentError):
    """LLM API errors (rate limit, invalid response)"""
    def __init__(self, message: str, model: str = None,
                 error_code: str = None, is_retryable: bool = False):
        super().__init__(message)
        self.model = model
        self.error_code = error_code
        self.is_retryable = is_retryable

class DataValidationError(AmoreAgentError):
    """Data validation errors"""
    def __init__(self, message: str, field: str = None,
                 value: Any = None, constraint: str = None):
        super().__init__(message)
        self.field = field
        self.value = value
        self.constraint = constraint

class ScraperError(AmoreAgentError):
    """Amazon scraping errors"""
    def __init__(self, message: str, category: str = None,
                 asin: str = None, error_type: str = None):
        super().__init__(message)
        self.category = category
        self.asin = asin
        self.error_type = error_type  # BLOCKED, TIMEOUT, PARSE_ERROR
```

### 적용 파일
- `src/agents/hybrid_insight_agent.py:272,435,687`
- `src/agents/crawler_agent.py:92-184`
- `src/api/routes/chat.py:460,655,723`
- `src/infrastructure/persistence/json_repository.py:95,134,139`

### 수용 기준
- [ ] 모든 예외 타입에 대한 테스트 통과
- [ ] 기존 `except Exception`을 구체적 예외로 교체
- [ ] 예외 발생 시 적절한 로깅

---

## 🎯 Phase 2: 에이전트 테스트 추가 (TDD)

### 2.1 HybridInsightAgent 테스트

#### 테스트 먼저 (RED)
```python
# tests/unit/agents/test_hybrid_insight_agent.py

class TestHybridInsightAgent:

    @pytest.fixture
    def mock_kg(self):
        """Mock KnowledgeGraph"""

    @pytest.fixture
    def mock_reasoner(self):
        """Mock OntologyReasoner"""

    @pytest.fixture
    def agent(self, mock_kg, mock_reasoner):
        """Agent with mocked dependencies"""

    # 정상 케이스
    async def test_generate_insight_returns_insight_result(self, agent):
        """generate_insight()는 InsightResult 반환해야 함"""

    async def test_update_kg_from_crawl_data(self, agent, mock_kg):
        """크롤 데이터로 KG 업데이트 확인"""

    async def test_update_kg_from_metrics_data(self, agent, mock_kg):
        """메트릭 데이터로 KG 업데이트 확인"""

    # 에러 케이스
    async def test_llm_timeout_raises_llm_api_error(self, agent):
        """LLM 타임아웃 시 LLMAPIError 발생"""

    async def test_empty_crawl_data_raises_validation_error(self, agent):
        """빈 크롤 데이터 시 DataValidationError 발생"""
```

### 2.2 HybridChatbotAgent 테스트

#### 테스트 먼저 (RED)
```python
# tests/unit/agents/test_hybrid_chatbot_agent.py

class TestHybridChatbotAgent:

    @pytest.fixture
    def agent(self, mock_retriever):
        """Agent with mocked HybridRetriever"""

    # 정상 케이스
    async def test_process_query_returns_response(self, agent):
        """process_query()는 ChatResponse 반환해야 함"""

    async def test_query_uses_hybrid_retrieval(self, agent, mock_retriever):
        """쿼리 시 HybridRetriever.retrieve() 호출 확인"""

    async def test_response_includes_sources(self, agent):
        """응답에 sources 포함 확인"""

    # 에러 케이스
    async def test_empty_query_raises_validation_error(self, agent):
        """빈 쿼리 시 DataValidationError 발생"""

    async def test_retriever_failure_graceful_degradation(self, agent):
        """Retriever 실패 시 graceful degradation"""
```

### 2.3 CrawlerAgent 테스트

#### 테스트 먼저 (RED)
```python
# tests/unit/agents/test_crawler_agent.py

class TestCrawlerAgent:

    @pytest.fixture
    def mock_scraper(self):
        """Mock AmazonScraper"""

    @pytest.fixture
    def agent(self, mock_scraper):
        """Agent with mocked scraper"""

    # 정상 케이스
    async def test_crawl_category_returns_products(self, agent):
        """crawl_category()는 Product 리스트 반환"""

    async def test_crawl_all_categories_parallel(self, agent):
        """모든 카테고리 크롤링 (병렬성 확인)"""

    async def test_product_brand_extraction(self, agent):
        """브랜드 추출 정확성 테스트"""

    # 에러 케이스
    async def test_blocked_raises_scraper_error(self, agent):
        """차단 시 ScraperError(error_type='BLOCKED') 발생"""

    async def test_timeout_raises_scraper_error(self, agent):
        """타임아웃 시 ScraperError(error_type='TIMEOUT') 발생"""

    async def test_partial_failure_returns_successful_categories(self, agent):
        """일부 카테고리 실패 시 성공한 카테고리만 반환"""
```

### 수용 기준
- [ ] 각 에이전트 테스트 파일 생성
- [ ] 정상/에러 케이스 모두 커버
- [ ] Mock을 통한 의존성 격리
- [ ] 테스트 실행 시 외부 API 호출 없음

---

## 🎯 Phase 3: 입력 검증 강화 (TDD)

### 목표
프롬프트 인젝션 방어 및 입력 살균

### 테스트 먼저 (RED)
```python
# tests/unit/api/test_input_validation.py

class TestInputValidator:

    def test_detects_prompt_injection_ignore_instructions(self):
        """'ignore previous instructions' 패턴 탐지"""

    def test_detects_prompt_injection_system_override(self):
        """'system:' 또는 'SYSTEM:' 패턴 탐지"""

    def test_detects_prompt_injection_jailbreak(self):
        """일반적인 jailbreak 패턴 탐지"""

    def test_sanitizes_html_tags(self):
        """HTML 태그 제거"""

    def test_enforces_max_length(self):
        """최대 길이 제한 (2000자)"""

    def test_allows_normal_korean_input(self):
        """정상 한글 입력 허용"""

    def test_allows_normal_english_input(self):
        """정상 영어 입력 허용"""

    def test_allows_brand_names_with_special_chars(self):
        """'e.l.f.', 'L'Oreal' 등 브랜드명 허용"""
```

### 구현 (GREEN)
```python
# src/api/validators/input_validator.py

import re
from typing import Tuple
from src.domain.exceptions import DataValidationError

class InputValidator:
    MAX_LENGTH = 2000

    INJECTION_PATTERNS = [
        r'ignore\s+(all\s+)?previous\s+instructions',
        r'disregard\s+(all\s+)?above',
        r'^system\s*:',
        r'you\s+are\s+now\s+',
        r'pretend\s+to\s+be',
        r'act\s+as\s+if',
        r'forget\s+everything',
        r'new\s+instructions\s*:',
    ]

    def validate(self, text: str) -> Tuple[bool, str]:
        """
        입력 텍스트 검증
        Returns: (is_valid, sanitized_or_error_message)
        """
        # 1. 길이 제한
        if len(text) > self.MAX_LENGTH:
            raise DataValidationError(
                f"Input exceeds {self.MAX_LENGTH} characters",
                field="message",
                value=len(text),
                constraint=f"max_length={self.MAX_LENGTH}"
            )

        # 2. 프롬프트 인젝션 탐지
        text_lower = text.lower()
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                raise DataValidationError(
                    "Potential prompt injection detected",
                    field="message",
                    value=text[:50],
                    constraint="no_injection_patterns"
                )

        # 3. HTML 살균 (태그 제거)
        sanitized = re.sub(r'<[^>]+>', '', text)

        return True, sanitized.strip()
```

### 적용 위치
- `src/api/routes/chat.py:302-320`

### 수용 기준
- [ ] 모든 인젝션 패턴 탐지 테스트 통과
- [ ] 정상 입력 허용 테스트 통과
- [ ] chat.py에 InputValidator 적용
- [ ] 탐지 시 적절한 에러 응답 반환

---

## 🎯 Phase 4: DI 컨테이너 구현 (TDD)

### 목표
핵심 에이전트 3개에 대한 DI 컨테이너 구현

### 테스트 먼저 (RED)
```python
# tests/unit/infrastructure/test_container.py

class TestContainer:

    def test_get_knowledge_graph_singleton(self):
        """KnowledgeGraph는 싱글톤이어야 함"""
        kg1 = Container.get_knowledge_graph()
        kg2 = Container.get_knowledge_graph()
        assert kg1 is kg2

    def test_get_insight_agent_with_dependencies(self):
        """InsightAgent는 주입된 의존성 사용"""
        agent = Container.get_insight_agent()
        assert agent.kg is Container.get_knowledge_graph()

    def test_get_chatbot_agent_with_dependencies(self):
        """ChatbotAgent는 주입된 의존성 사용"""

    def test_get_crawler_agent_with_dependencies(self):
        """CrawlerAgent는 주입된 의존성 사용"""

    def test_reset_clears_all_instances(self):
        """reset()은 모든 인스턴스 초기화"""
        Container.get_knowledge_graph()
        Container.reset()
        assert Container._kg is None

    def test_override_for_testing(self):
        """테스트용 Mock 주입 가능"""
        mock_kg = MagicMock()
        Container.override('knowledge_graph', mock_kg)
        assert Container.get_knowledge_graph() is mock_kg
```

### 구현 (GREEN)
```python
# src/infrastructure/container.py

from typing import Optional, Dict, Any
from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import OntologyReasoner
from src.rag.hybrid_retriever import HybridRetriever
from src.agents.hybrid_insight_agent import HybridInsightAgent
from src.agents.hybrid_chatbot_agent import HybridChatbotAgent
from src.agents.crawler_agent import CrawlerAgent

class Container:
    """Dependency Injection Container for AMORE Agent"""

    _instances: Dict[str, Any] = {}
    _overrides: Dict[str, Any] = {}

    @classmethod
    def get_knowledge_graph(cls) -> KnowledgeGraph:
        if 'knowledge_graph' in cls._overrides:
            return cls._overrides['knowledge_graph']
        if 'knowledge_graph' not in cls._instances:
            cls._instances['knowledge_graph'] = KnowledgeGraph()
        return cls._instances['knowledge_graph']

    @classmethod
    def get_reasoner(cls) -> OntologyReasoner:
        if 'reasoner' in cls._overrides:
            return cls._overrides['reasoner']
        if 'reasoner' not in cls._instances:
            cls._instances['reasoner'] = OntologyReasoner(cls.get_knowledge_graph())
        return cls._instances['reasoner']

    @classmethod
    def get_hybrid_retriever(cls) -> HybridRetriever:
        if 'retriever' in cls._overrides:
            return cls._overrides['retriever']
        if 'retriever' not in cls._instances:
            cls._instances['retriever'] = HybridRetriever(
                knowledge_graph=cls.get_knowledge_graph(),
                reasoner=cls.get_reasoner()
            )
        return cls._instances['retriever']

    @classmethod
    def get_insight_agent(cls) -> HybridInsightAgent:
        if 'insight_agent' in cls._overrides:
            return cls._overrides['insight_agent']
        return HybridInsightAgent(
            knowledge_graph=cls.get_knowledge_graph(),
            reasoner=cls.get_reasoner(),
            retriever=cls.get_hybrid_retriever()
        )

    @classmethod
    def get_chatbot_agent(cls) -> HybridChatbotAgent:
        if 'chatbot_agent' in cls._overrides:
            return cls._overrides['chatbot_agent']
        return HybridChatbotAgent(
            retriever=cls.get_hybrid_retriever()
        )

    @classmethod
    def get_crawler_agent(cls) -> CrawlerAgent:
        if 'crawler_agent' in cls._overrides:
            return cls._overrides['crawler_agent']
        return CrawlerAgent()

    @classmethod
    def override(cls, name: str, instance: Any) -> None:
        """테스트용 Mock 주입"""
        cls._overrides[name] = instance

    @classmethod
    def reset(cls) -> None:
        """모든 인스턴스 초기화"""
        cls._instances.clear()
        cls._overrides.clear()
```

### 수용 기준
- [ ] 모든 컨테이너 테스트 통과
- [ ] 싱글톤 동작 확인
- [ ] Mock 주입 가능 확인
- [ ] 기존 코드에서 Container 사용으로 전환

---

## 🎯 Phase 5: 커버리지 검증

### 실행 명령
```bash
# 전체 테스트 실행 + 커버리지
pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# 커버리지 리포트 확인
open htmlcov/index.html
```

### 수용 기준
- [ ] 전체 커버리지 70% 이상
- [ ] 핵심 모듈 커버리지:
  - `src/agents/`: 80% 이상
  - `src/domain/exceptions.py`: 100%
  - `src/api/validators/`: 90% 이상
  - `src/infrastructure/container.py`: 90% 이상

---

## 📁 생성될 파일 목록

### 새 파일
```
src/domain/exceptions.py                      # 커스텀 예외 타입
src/api/validators/input_validator.py         # 입력 검증기
src/infrastructure/container.py               # DI 컨테이너
tests/unit/domain/test_exceptions.py          # 예외 테스트
tests/unit/agents/test_hybrid_insight_agent.py
tests/unit/agents/test_hybrid_chatbot_agent.py
tests/unit/agents/test_crawler_agent.py
tests/unit/api/test_input_validation.py
tests/unit/infrastructure/test_container.py
```

### 수정될 파일
```
src/agents/hybrid_insight_agent.py            # 예외 처리 + DI 적용
src/agents/hybrid_chatbot_agent.py            # 예외 처리 + DI 적용
src/agents/crawler_agent.py                   # 예외 처리 + DI 적용
src/api/routes/chat.py                        # 입력 검증 적용
src/infrastructure/persistence/json_repository.py  # 예외 처리
```

---

## ⏱️ 실행 순서

1. **Phase 1** 시작: `test_exceptions.py` 작성 (RED)
2. **Phase 1** 구현: `exceptions.py` 작성 (GREEN)
3. **Phase 1** 적용: 기존 파일에 예외 적용 (REFACTOR)
4. **Phase 2** 시작: 에이전트 테스트 작성 (RED)
5. **Phase 2** 통과: Mock 기반 테스트 통과 (GREEN)
6. **Phase 3** 시작: `test_input_validation.py` 작성 (RED)
7. **Phase 3** 구현: `input_validator.py` 작성 (GREEN)
8. **Phase 3** 적용: `chat.py`에 적용 (REFACTOR)
9. **Phase 4** 시작: `test_container.py` 작성 (RED)
10. **Phase 4** 구현: `container.py` 작성 (GREEN)
11. **Phase 4** 적용: 에이전트에 DI 적용 (REFACTOR)
12. **Phase 5**: 커버리지 70% 확인

---

## ✅ 전체 수용 기준 체크리스트

- [ ] Phase 1: 커스텀 예외 4종 정의 및 테스트 통과
- [ ] Phase 2: 에이전트 테스트 3개 파일 생성 및 통과
- [ ] Phase 3: 입력 검증기 테스트 통과 및 chat.py 적용
- [ ] Phase 4: DI 컨테이너 테스트 통과 및 에이전트 적용
- [ ] Phase 5: 전체 커버리지 70% 달성
- [ ] 모든 기존 테스트 통과 (`pytest tests/ -v`)
