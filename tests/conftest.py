import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from dotenv import load_dotenv

_CHROMA_TMP_DIR: str | None = None


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

    # === Track 0-F: ChromaDB 쓰기 격리 (전역 안전망) ===
    # src/rag/retriever.py는 `os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")`로
    # 컬렉션 경로를 읽으므로, 실제 data/chroma/를 세션 임시 디렉토리로 복사한 뒤
    # 이 환경변수로 리다이렉트한다. 실제 인덱싱된 문서(358+82개 임베딩)를 그대로
    # 복사하므로 검색 결과를 검증하는 기존 테스트는 그대로 통과하고, chromadb가
    # 열람 시 남기는 내부 부기(sqlite WAL/HNSW 메타데이터 등) 쓰기만 임시 경로로
    # 격리되어 실제 data/chroma/는 전혀 건드리지 않는다.
    import shutil
    import tempfile

    global _CHROMA_TMP_DIR
    real_chroma_dir = project_root / "data" / "chroma"
    if real_chroma_dir.exists() and "CHROMA_PERSIST_DIR" not in os.environ:
        _CHROMA_TMP_DIR = tempfile.mkdtemp(prefix="amore-test-chroma-")
        shutil.copytree(real_chroma_dir, _CHROMA_TMP_DIR, dirs_exist_ok=True)
        os.environ["CHROMA_PERSIST_DIR"] = _CHROMA_TMP_DIR
        print(f"[conftest] CHROMA_PERSIST_DIR isolated to: {_CHROMA_TMP_DIR}")


def pytest_unconfigure(config):
    """테스트 세션 종료 시 임시 Chroma 복사본 정리"""
    import shutil

    if _CHROMA_TMP_DIR:
        shutil.rmtree(_CHROMA_TMP_DIR, ignore_errors=True)


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
