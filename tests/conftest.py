import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from dotenv import load_dotenv


def pytest_configure(config):
    """테스트 시작 전 환경 설정 로드"""
    project_root = Path(__file__).parent.parent

    main_env_path = project_root / ".env"
    if main_env_path.exists():
        load_dotenv(main_env_path, override=False)
        print(f"\n[conftest] Loaded base environment from: {main_env_path}")

    env_file = os.environ.get("ENV_FILE", ".env.test")
    env_path = project_root / env_file

    if env_path.exists():
        load_dotenv(env_path, override=True)
        print(f"[conftest] Applied test overrides from: {env_path}")
    else:
        print(f"[conftest] No {env_file} found, using base environment only")

    # === F15: 실제 OpenAI API 호출 차단 가드 ===
    # .env에 실제 키가 로드되어 있어도, 테스트 세션 동안에는 절대 실제 OpenAI로
    # 나가지 못하도록 API 키를 더미 값으로, base URL을 닫힌 로컬 주소로 강제
    # override한다. litellm은 OPENAI_BASE_URL을 OPENAI_API_BASE보다 우선 사용하므로
    # (litellm/llms/openai/common_utils.py 등) 둘 다 닫아둔다.
    # 새는 호출은 로컬의 열려 있지 않은 포트로 연결을 시도하다 즉시
    # 연결 거부(connection refused)로 실패한다 — 과금 없이, 네트워크 대기 없이.
    os.environ["OPENAI_API_KEY"] = (
        "sk-test-dummy-guard-0000000000000000000000"  # pragma: allowlist secret
    )
    os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9/v1"
    os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:9/v1"
    print(
        "[conftest] OpenAI API guard active: OPENAI_API_KEY/OPENAI_BASE_URL/OPENAI_API_BASE "
        "forced to dummy/closed values for this test session"
    )


@pytest.fixture
def results():
    """테스트 결과 객체 (인라인 정의, 외부 의존 제거)"""

    class _TestResult:
        def __init__(self):
            self.passed = 0
            self.failed = 0
            self.errors = []

        def record_pass(self, test_name: str):
            self.passed += 1
            print(f"  ✅ {test_name}")

        def record_fail(self, test_name: str, error: str):
            self.failed += 1
            self.errors.append(f"{test_name}: {error}")
            print(f"  ❌ {test_name}: {error}")

        def summary(self) -> str:
            total = self.passed + self.failed
            return f"\n{'=' * 60}\n총 {total}개 테스트: ✅ {self.passed} 성공, ❌ {self.failed} 실패\n{'=' * 60}"

    return _TestResult()


@pytest_asyncio.fixture
async def retriever():
    """Mock retriever - 실제 ChromaDB/디스크 의존성 제거"""
    mock = MagicMock()
    mock.initialize = AsyncMock()
    mock.retrieve = AsyncMock(return_value=[])
    mock.get_relevant_documents = AsyncMock(return_value=[])
    return mock
