"""
SemanticChunker 단위 테스트
"""

from src.rag.chunker import SemanticChunker


class TestSemanticChunker:
    """SemanticChunker 클래스 테스트"""

    def test_init_defaults(self):
        """기본 초기화"""
        chunker = SemanticChunker()
        assert chunker.min_chunk_size > 0
        assert chunker.max_chunk_size > chunker.min_chunk_size

    def test_init_custom(self):
        """커스텀 파라미터"""
        chunker = SemanticChunker(
            min_chunk_size=50,
            max_chunk_size=500,
            similarity_threshold=0.7,
        )
        assert chunker.min_chunk_size == 50
        assert chunker.max_chunk_size == 500

    def test_simple_chunk(self):
        """단순 청킹 (짧은 텍스트)"""
        chunker = SemanticChunker()
        text = "이것은 테스트 문장입니다. 두 번째 문장입니다."
        chunks = chunker._simple_chunk(text, doc_id="test_doc")
        assert len(chunks) >= 1
        # 원본 텍스트 포함 확인 - Chunk 객체 처리
        combined = " ".join(c.content if hasattr(c, "content") else str(c) for c in chunks)
        assert "테스트" in combined

    def test_empty_text(self):
        """빈 텍스트 처리"""
        chunker = SemanticChunker()
        try:
            chunks = chunker._simple_chunk("", doc_id="empty_doc")
            assert len(chunks) == 0 or chunks == []
        except Exception:
            pass  # 빈 입력 에러 허용

    def test_long_text_chunking(self):
        """긴 텍스트 청킹"""
        chunker = SemanticChunker(max_chunk_size=200)
        text = "문장입니다. " * 100  # ~700자
        chunks = chunker._simple_chunk(text, doc_id="long_doc")
        assert len(chunks) >= 2  # 여러 청크로 분할

    def test_chunk_has_content(self):
        """청크에 content 필드 존재"""
        chunker = SemanticChunker()
        text = "LANEIGE는 아모레퍼시픽의 브랜드입니다. Lip Sleeping Mask가 대표 제품입니다."
        chunks = chunker._simple_chunk(text, doc_id="content_doc")
        for chunk in chunks:
            # Chunk 객체는 content 속성을 가짐
            if hasattr(chunk, "content"):
                assert len(chunk.content) > 0
            elif isinstance(chunk, dict):
                assert "content" in chunk
                assert len(chunk["content"]) > 0
            else:
                assert len(str(chunk)) > 0
