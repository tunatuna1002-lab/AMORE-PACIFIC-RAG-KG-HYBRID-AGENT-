import pytest
import pytest_asyncio

from tests.test_rag_integration import TestResult
from src.rag.retriever import DocumentRetriever


@pytest.fixture
def results():
    return TestResult()


@pytest_asyncio.fixture
async def retriever():
    retriever = DocumentRetriever(docs_path="./docs")
    await retriever.initialize()
    return retriever
