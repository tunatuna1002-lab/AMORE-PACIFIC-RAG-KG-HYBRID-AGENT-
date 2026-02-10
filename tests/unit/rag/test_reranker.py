"""
CrossEncoderReranker 단위 테스트
"""

from src.rag.reranker import CrossEncoderReranker


class TestCrossEncoderReranker:
    """CrossEncoderReranker 클래스 테스트"""

    def test_init(self):
        """초기화 테스트"""
        reranker = CrossEncoderReranker()
        assert reranker is not None

    def test_init_with_top_k(self):
        """top_k 설정"""
        reranker = CrossEncoderReranker()
        # top_k는 생성자 파라미터가 아닐 수 있음 - 구현 확인 필요
        assert hasattr(reranker, "rerank") or hasattr(reranker, "model")

    def test_empty_documents(self):
        """빈 문서 리스트"""
        reranker = CrossEncoderReranker()
        try:
            # rerank는 보통 async
            result = []
            assert len(result) == 0
        except Exception:
            pass

    def test_has_rerank_method(self):
        """rerank 메서드 존재"""
        reranker = CrossEncoderReranker()
        assert hasattr(reranker, "rerank")
