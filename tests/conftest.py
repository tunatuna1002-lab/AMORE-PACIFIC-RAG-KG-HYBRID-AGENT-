import os
from pathlib import Path

import pytest
import pytest_asyncio
from dotenv import load_dotenv

from tests.test_rag_integration import TestResult
from src.rag.retriever import DocumentRetriever


def pytest_configure(config):
    """테스트 시작 전 환경 설정 로드"""
    env_file = os.environ.get("ENV_FILE", ".env.test")
    project_root = Path(__file__).parent.parent
    env_path = project_root / env_file

    if env_path.exists():
        load_dotenv(env_path, override=True)
        print(f"\n[conftest] Loaded environment from: {env_path}")
    else:
        print(f"\n[conftest] No {env_file} found, using default environment")


@pytest.fixture
def results():
    return TestResult()


@pytest_asyncio.fixture
async def retriever():
    retriever = DocumentRetriever(docs_path="./docs")
    await retriever.initialize()
    return retriever
