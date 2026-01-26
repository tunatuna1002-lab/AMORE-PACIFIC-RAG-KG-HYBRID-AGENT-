import os
from pathlib import Path

import pytest
import pytest_asyncio
from dotenv import load_dotenv

from tests.test_rag_integration import TestResult
from src.rag.retriever import DocumentRetriever


def pytest_configure(config):
    """테스트 시작 전 환경 설정 로드"""
    project_root = Path(__file__).parent.parent

    # 1. 먼저 .env 로드 (기본 API 키들)
    main_env_path = project_root / ".env"
    if main_env_path.exists():
        load_dotenv(main_env_path, override=False)
        print(f"\n[conftest] Loaded base environment from: {main_env_path}")

    # 2. 테스트 전용 설정 덮어쓰기 (.env.test)
    env_file = os.environ.get("ENV_FILE", ".env.test")
    env_path = project_root / env_file

    if env_path.exists():
        load_dotenv(env_path, override=True)
        print(f"[conftest] Applied test overrides from: {env_path}")
    else:
        print(f"[conftest] No {env_file} found, using base environment only")


@pytest.fixture
def results():
    return TestResult()


@pytest_asyncio.fixture
async def retriever():
    retriever = DocumentRetriever(docs_path="./docs")
    await retriever.initialize()
    return retriever
