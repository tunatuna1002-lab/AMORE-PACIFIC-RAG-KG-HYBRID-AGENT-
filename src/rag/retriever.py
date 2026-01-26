"""
Document Retriever
RAG를 위한 문서 검색 모듈

14개 MD 파일 기반:

[Type D: 기존 지표 가이드] - docs/guides/
- Strategic Indicators Definition.md
- Metric Interpretation Guide.md
- Indicator Combination Playbook.md
- Home Page Insight Rules.md

[Type A: 분석 플레이북] - docs/market/
- 아마존 랭킹 급등 원인 역추적 보고서.md
- 아마존 랭킹 변동 원인 분석 가이드.md

[Type B: 시장 인텔리전스] - docs/market/
- (1) K-뷰티 초격차의 서막.md
- 미국 뷰티 트렌드 레이더.md
- 뷰티 트렌드 분석 및 판매 전략 제안.md

[Type C: 대응 가이드] - docs/market/
- 부정 이슈 조기경보 및 대응 프롬프트.md
- 인플루언서 맵 & 메시지 맵 생성.md

[Type E: IR 분기 실적 보고서] - docs/ir/
- AP_1Q25_EN.md (아모레퍼시픽 2025 Q1)
- AP_2Q25_EN.md (아모레퍼시픽 2025 Q2)
- AP_3Q25_EN.md (아모레퍼시픽 2025 Q3)
"""

import os
import re
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from .reranker import get_reranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False

try:
    from .chunker import get_semantic_chunker
    SEMANTIC_CHUNKER_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKER_AVAILABLE = False

# Lazy import flag - 실제 사용 시 초기화
VECTOR_SEARCH_AVAILABLE = None


class DocumentRetriever:
    """문서 검색 클래스 (TTL 캐싱 지원)"""

    # 설정 파일 경로
    CONFIG_PATH = "config/thresholds.json"

    # 검색 결과 캐시 (maxsize=100, TTL=5분)
    _search_cache: Dict[str, Any] = {}
    _cache_timestamps: Dict[str, float] = {}
    _CACHE_TTL = None  # 설정에서 로드

    @classmethod
    def _load_config(cls) -> dict:
        """설정 파일에서 RAG 관련 설정 로드"""
        import json
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / cls.CONFIG_PATH

        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config.get("system", {}).get("rag", {})
            except Exception:
                pass

        return {}  # 설정 없으면 기본값 사용

    @classmethod
    def get_cache_ttl(cls) -> int:
        """캐시 TTL 반환 (설정 파일에서 로드, 기본 300초)"""
        if cls._CACHE_TTL is None:
            config = cls._load_config()
            cls._CACHE_TTL = config.get("ttl_seconds", 300)
        return cls._CACHE_TTL

    # 문서 메타데이터
    DOCUMENTS = {
        # ========== Type D: 기존 지표 가이드 (docs/guides/) ==========
        "strategic_indicators": {
            "filename": "Strategic Indicators Definition.md",
            "description": "지표 정의 및 산출식",
            "doc_type": "metric_guide",
            "keywords": ["정의", "산출식", "SoS", "HHI", "CPI", "계산", "공식"],
            "intent_triggers": ["정의", "공식", "계산", "산출"],
            "freshness": "static"
        },
        "metric_interpretation": {
            "filename": "Metric Interpretation Guide.md",
            "description": "지표 해석 가이드",
            "doc_type": "metric_guide",
            "keywords": ["해석", "의미", "높음", "낮음", "주의사항", "함께 봐야"],
            "intent_triggers": ["의미", "해석", "뜻"],
            "freshness": "static"
        },
        "indicator_combination": {
            "filename": "Indicator Combination Playbook.md",
            "description": "지표 조합 해석 플레이북",
            "doc_type": "metric_guide",
            "keywords": ["조합", "시나리오", "액션", "전략", "상승", "하락"],
            "intent_triggers": ["조합", "같이", "함께", "시나리오"],
            "freshness": "static"
        },
        "home_insight_rules": {
            "filename": "Home Page Insight Rules.md",
            "description": "인사이트 생성 규칙",
            "doc_type": "metric_guide",
            "keywords": ["인사이트", "요약", "문구", "템플릿", "톤", "안전장치"],
            "intent_triggers": ["인사이트", "요약", "규칙"],
            "freshness": "static"
        },
        
        # ========== Type A: 분석 플레이북 (docs/market/) ==========
        "amazon_ranking_diagnosis": {
            "filename": "아마존 랭킹 급등 원인 역추적 보고서.md",
            "description": "BSR 급변 원인 진단 체크리스트 및 If-Then 가설 트리",
            "doc_type": "playbook",
            "keywords": ["순위", "BSR", "급등", "급락", "원인", "분석", "체크리스트",
                        "가설", "재고", "광고", "프로모션", "리뷰", "가격"],
            "intent_triggers": ["왜", "원인", "갑자기", "급변", "떨어", "올라", "변동"],
            "freshness": "quarterly"
        },
        "amazon_algorithm_guide": {
            "filename": "아마존 랭킹 변동 원인 분석 가이드.md",
            "description": "COSMO/Rufus 알고리즘 대응 및 심층 진단",
            "doc_type": "playbook",
            "keywords": ["알고리즘", "COSMO", "Rufus", "A10", "검색", "억제",
                        "외부트래픽", "틱톡", "바이럴", "지식그래프", "BSR"],
            "intent_triggers": ["알고리즘", "검색", "노출", "억제", "틱톡", "자세히"],
            "freshness": "quarterly"
        },
        
        # ========== Type B: 시장 인텔리전스 (docs/market/) ==========
        "kbeauty_industry": {
            "filename": "(1) K-뷰티 초격차의 서막 [풀영상] _ 창 534회 (KBS 26.1.20.) - YouTube.md",
            "description": "K-뷰티 산업 배경 (ODM, 글로벌 확장, 중국 위협)",
            "doc_type": "knowledge_base",
            "keywords": ["K-뷰티", "ODM", "글로벌", "중국", "미용기기", "맞춤화장품",
                        "콘텐츠", "편집숍", "아마존", "초격차", "한국 화장품"],
            "intent_triggers": ["K-뷰티", "한국 화장품", "산업", "배경", "ODM"],
            "freshness": "static"
        },
        "us_beauty_trends_weekly": {
            "filename": "미국 뷰티 트렌드 레이더.md",
            "description": "미국 주간 뷰티 트렌드 Top 10 및 LANEIGE 연결 가설",
            "doc_type": "intelligence",
            "keywords": ["트렌드", "펩타이드", "PDRN", "립케어", "글래스스킨",
                        "세라마이드", "스네일뮤신", "나이아신아마이드", "키워드", "TikTok"],
            "intent_triggers": ["트렌드", "요즘", "최근", "인기", "바이럴", "키워드"],
            "freshness": "weekly",
            "valid_period": "2025-12-21 ~ 2026-01-20"
        },
        "laneige_strategy_2026": {
            "filename": "뷰티 트렌드 분석 및 판매 전략 제안.md",
            "description": "2026년 1월 LANEIGE 아마존 판매 전략 (모닝쉐드, PDRN, 립케어)",
            "doc_type": "intelligence",
            "keywords": ["전략", "판매", "모닝쉐드", "슬리핑마스크", "번들",
                        "립베이스팅", "핑크펩타이드", "워터뱅크", "크림스킨", "LANEIGE"],
            "intent_triggers": ["전략", "어떻게", "제안", "추천", "LANEIGE"],
            "freshness": "monthly",
            "target_brand": "laneige"
        },
        
        # ========== Type C: 대응 가이드 (docs/market/) ==========
        "negative_issue_response": {
            "filename": "부정 이슈 조기경보 및 대응 프롬프트.md",
            "description": "브랜드별 부정 이슈 분석 및 대응 문구 (라운드랩, 아누아, 티르티르)",
            "doc_type": "response_guide",
            "keywords": ["부정", "위기", "리뷰", "대응", "라운드랩", "아누아",
                        "티르티르", "가품", "리포뮬레이션", "끈적임", "산화", "트러블"],
            "intent_triggers": ["부정", "문제", "이슈", "대응", "어떻게 해", "위기"],
            "freshness": "monthly",
            "brands_covered": ["round_lab", "anua", "tirtir", "beef_tallow"]
        },
        "laneige_influencer_map": {
            "filename": "인플루언서 맵 & 메시지 맵 생성.md",
            "description": "LANEIGE 채널별 인플루언서 분류 및 크리에이티브 훅 5선",
            "doc_type": "response_guide",
            "keywords": ["인플루언서", "틱톡", "유튜브", "레딧", "인스타그램",
                        "메시지", "크리에이티브", "훅", "리스크", "LANEIGE", "마케팅"],
            "intent_triggers": ["인플루언서", "마케팅", "메시지", "콘텐츠", "크리에이터"],
            "freshness": "monthly",
            "target_brand": "laneige"
        },

        # ========== Type E: IR 분기 실적 보고서 (docs/ir/) ==========
        "ir_2025_q1": {
            "filename": "AP_1Q25_EN.md",
            "description": "아모레퍼시픽 2025 Q1 실적 (COSRX 편입, Americas +102%)",
            "doc_type": "ir_report",
            "keywords": ["매출", "영업이익", "Revenue", "Operating Profit",
                        "Americas", "COSRX", "LANEIGE", "Sulwhasoo", "Prime Day",
                        "Western Region", "Greater China", "Q1", "1분기", "실적",
                        "아모레퍼시픽", "Amorepacific", "IR", "earnings"],
            "intent_triggers": ["Q1", "1분기", "2025", "실적", "매출", "Americas",
                               "COSRX 편입", "IR", "분기", "earnings"],
            "freshness": "quarterly",
            "quarter": "2025-Q1",
            "parent_company": "amorepacific"
        },
        "ir_2025_q2": {
            "filename": "AP_2Q25_EN.md",
            "description": "아모레퍼시픽 2025 Q2 실적 (Greater China 턴어라운드, OP +1673%)",
            "doc_type": "ir_report",
            "keywords": ["매출", "영업이익", "Revenue", "Operating Profit",
                        "Americas", "Greater China", "턴어라운드", "LANEIGE",
                        "Neo Cushion", "Aestura", "Q2", "2분기", "실적",
                        "아모레퍼시픽", "Amorepacific", "IR", "earnings", "중국"],
            "intent_triggers": ["Q2", "2분기", "2025", "중국", "Greater China",
                               "실적", "IR", "분기", "earnings", "턴어라운드"],
            "freshness": "quarterly",
            "quarter": "2025-Q2",
            "parent_company": "amorepacific"
        },
        "ir_2025_q3": {
            "filename": "AP_3Q25_EN.md",
            "description": "아모레퍼시픽 2025 Q3 실적 (Prime Day 2배, Americas +6.9%)",
            "doc_type": "ir_report",
            "keywords": ["매출", "영업이익", "Revenue", "Operating Profit",
                        "Americas", "Prime Day", "아마존", "Amazon", "LANEIGE",
                        "Illiyoon", "Mise-en-scène", "Q3", "3분기", "실적",
                        "아모레퍼시픽", "Amorepacific", "IR", "earnings"],
            "intent_triggers": ["Q3", "3분기", "2025", "Prime Day", "아마존",
                               "실적", "IR", "분기", "earnings", "최근"],
            "freshness": "quarterly",
            "quarter": "2025-Q3",
            "parent_company": "amorepacific"
        }
    }

    def __init__(
        self,
        docs_path: str = "./docs",
        use_semantic_chunking: bool = False,
        use_reranker: bool = True,
        use_query_expansion: bool = True
    ):
        """
        Args:
            docs_path: 문서 폴더 경로
            use_semantic_chunking: Semantic Chunker 사용 여부
            use_reranker: Reranker 사용 여부 (기본값: True)
            use_query_expansion: Query Expansion 사용 여부 (기본값: True)
        """
        self.docs_path = Path(docs_path)
        self.documents: Dict[str, str] = {}
        self.chunks: List[Dict[str, Any]] = []
        self._chunk_index: Dict[str, Dict[str, Any]] = {}
        self.embedding_model = None
        self.openai_client = None
        self.embedding_model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.collection = None
        self._initialized = False

        # 새 기능 플래그
        self.use_semantic_chunking = use_semantic_chunking
        self.use_reranker = use_reranker and RERANKER_AVAILABLE
        self.use_query_expansion = use_query_expansion

        # Reranker 인스턴스 (lazy loading)
        self._reranker = None

    def _check_vector_search(self) -> bool:
        """벡터 검색 가능 여부 확인 (OpenAI Embeddings + ChromaDB)"""
        global VECTOR_SEARCH_AVAILABLE
        if VECTOR_SEARCH_AVAILABLE is None:
            try:
                import chromadb
                import openai

                api_key = os.getenv("OPENAI_API_KEY")
                VECTOR_SEARCH_AVAILABLE = bool(api_key)
            except ImportError:
                VECTOR_SEARCH_AVAILABLE = False
            except Exception:
                VECTOR_SEARCH_AVAILABLE = False
        return VECTOR_SEARCH_AVAILABLE

    async def initialize(self) -> bool:
        """
        문서 로드 및 벡터 인덱스 초기화

        Returns:
            초기화 성공 여부

        Raises:
            ValueError: 벡터 검색 초기화 실패 시
        """
        # 문서 로드
        await self._load_documents()

        # 벡터 검색 초기화 (필수)
        if not self._check_vector_search():
            raise ValueError(
                "Vector search is required but not available. "
                "Please install required packages: chromadb, openai"
            )

        await self._initialize_vector_search()

        if self.collection is None:
            raise ValueError("Vector search initialization failed")

        self._initialized = True
        return True

    async def _load_documents(self) -> None:
        """MD 문서 로드"""
        # 프로젝트 루트에서 MD 파일 찾기
        root_path = self.docs_path.parent
        guides_path = self.docs_path / "guides"  # docs/guides/ 폴더
        market_path = self.docs_path / "market"  # docs/market/ 폴더
        ir_path = self.docs_path / "ir"  # docs/ir/ 폴더 (IR 실적 보고서)

        # Semantic Chunker 초기화 (옵션)
        semantic_chunker = None
        if self.use_semantic_chunking and SEMANTIC_CHUNKER_AVAILABLE:
            semantic_chunker = get_semantic_chunker()

        for doc_id, doc_info in self.DOCUMENTS.items():
            # docs/guides, docs/market, docs/ir, docs, 루트 폴더 순으로 검색
            possible_paths = [
                guides_path / doc_info["filename"],  # docs/guides/
                market_path / doc_info["filename"],  # docs/market/
                ir_path / doc_info["filename"],  # docs/ir/ (IR 보고서)
                self.docs_path / doc_info["filename"],  # docs/
                root_path / doc_info["filename"]  # 프로젝트 루트
            ]

            for file_path in possible_paths:
                if file_path.exists():
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        self.documents[doc_id] = content

                        # 청크 분할
                        if semantic_chunker:
                            # Semantic Chunking 사용
                            chunks = semantic_chunker.chunk_document(content, doc_info)
                        else:
                            # 기존 청킹 방식
                            doc_type = doc_info.get("doc_type", "metric_guide")
                            chunk_size = self._get_chunk_size_by_type(doc_type)
                            chunks = self._split_into_chunks(content, doc_id, doc_info, chunk_size)

                        self.chunks.extend(chunks)
                    break

        # 청크 인덱스 갱신 (벡터 검색 결과 메타데이터 보강용)
        self._chunk_index = {chunk["id"]: chunk for chunk in self.chunks}
    
    def _get_chunk_size_by_type(self, doc_type: str) -> int:
        """문서 유형별 청크 크기 반환"""
        chunk_sizes = {
            "playbook": 800,      # Type A: 분석 플레이북 - 큰 청크
            "intelligence": 600,  # Type B: 시장 인텔리전스
            "knowledge_base": 600,  # Type B: 지식 베이스
            "response_guide": 500,  # Type C: 대응 가이드
            "metric_guide": 500,  # Type D: 기존 지표 가이드
            "ir_report": 700      # Type E: IR 분기 실적 - 테이블 포함 큰 청크
        }
        return chunk_sizes.get(doc_type, 500)

    def _split_into_chunks(
        self,
        content: str,
        doc_id: str,
        doc_info: Dict,
        chunk_size: int = 500
    ) -> List[Dict[str, Any]]:
        """
        문서를 청크로 분할
        
        - 표(Table)는 별도 청크로 분리하여 완전성 유지
        - 섹션 기반 분할 후 크기 초과 시 추가 분할
        """
        chunks = []
        doc_type = doc_info.get("doc_type", "metric_guide")
        source_filename = doc_info.get("filename", "")
        target_brand = doc_info.get("target_brand")
        brands_covered = doc_info.get("brands_covered", [])
        
        # 1. 표(Table) 추출 및 별도 청크 생성
        table_pattern = r'(\|[^\n]+\|\n(?:\|[-:| ]+\|\n)?(?:\|[^\n]+\|\n)+)'
        tables = re.findall(table_pattern, content)
        
        for t_idx, table in enumerate(tables):
            table_text = table.strip()
            if table_text:
                # 표 주변 컨텍스트 찾기 (표 바로 위의 제목)
                table_pos = content.find(table)
                context_before = content[:table_pos].strip()
                lines_before = context_before.split('\n')
                
                # 표 제목 추출 (### 또는 **로 시작하는 마지막 라인)
                table_title = ""
                for line in reversed(lines_before[-5:]):
                    line_stripped = line.strip()
                    if line_stripped.startswith('#') or line_stripped.startswith('**'):
                        table_title = line_stripped.replace('#', '').replace('*', '').strip()
                        break
                
                chunks.append({
                    "id": f"{doc_id}_table_{t_idx}",
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "title": table_title or f"Table {t_idx + 1}",
                    "content": table_text,
                    "content_type": "table",
                    "source_filename": source_filename,
                    "target_brand": target_brand,
                    "brands_covered": brands_covered,
                    "keywords": doc_info["keywords"],
                    "description": doc_info["description"]
                })
        
        # 2. 표를 플레이스홀더로 대체한 후 섹션 분할
        content_without_tables = re.sub(table_pattern, '\n[TABLE]\n', content)
        sections = content_without_tables.split("\n## ")

        for i, section in enumerate(sections):
            if not section.strip():
                continue
            
            # [TABLE] 플레이스홀더만 있는 섹션은 스킵
            if section.strip() == "[TABLE]":
                continue

            # 섹션 제목 추출
            lines = section.split("\n")
            title = lines[0].replace("#", "").strip() if lines else ""

            # 청크 생성
            text = section.strip()
            
            # [TABLE] 플레이스홀더 제거
            text = re.sub(r'\n*\[TABLE\]\n*', '\n', text).strip()
            
            if not text:
                continue
            
            if len(text) > chunk_size:
                # 긴 섹션은 추가 분할
                sub_chunks = self._smart_split(text, chunk_size)
                for k, sub_chunk in enumerate(sub_chunks):
                    if sub_chunk.strip():
                        chunks.append({
                            "id": f"{doc_id}_{i}_{k}",
                            "doc_id": doc_id,
                            "doc_type": doc_type,
                            "title": title,
                            "content": sub_chunk,
                            "content_type": "text",
                            "source_filename": source_filename,
                            "target_brand": target_brand,
                            "brands_covered": brands_covered,
                            "keywords": doc_info["keywords"],
                            "description": doc_info["description"]
                        })
            else:
                chunks.append({
                    "id": f"{doc_id}_{i}",
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "title": title,
                    "content": text,
                    "content_type": "text",
                    "source_filename": source_filename,
                    "target_brand": target_brand,
                    "brands_covered": brands_covered,
                    "keywords": doc_info["keywords"],
                    "description": doc_info["description"]
                })

        return chunks
    
    def _smart_split(self, text: str, chunk_size: int) -> List[str]:
        """
        텍스트를 의미 단위로 분할
        
        - 단락(\n\n) 기준으로 우선 분할
        - 단락이 chunk_size보다 크면 문장 단위로 분할
        """
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk = f"{current_chunk}\n\n{para}".strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                if len(para) <= chunk_size:
                    current_chunk = para
                else:
                    # 단락이 chunk_size보다 크면 강제 분할
                    for j in range(0, len(para), chunk_size):
                        chunks.append(para[j:j+chunk_size])
                    current_chunk = ""
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    async def _initialize_vector_search(self) -> None:
        """
        벡터 검색 초기화 (필수)

        Raises:
            ImportError: 필수 패키지 미설치 시
            ValueError: OPENAI_API_KEY 미설정 시
            Exception: 기타 초기화 실패 시
        """
        if not VECTOR_SEARCH_AVAILABLE:
            raise ImportError("Vector search dependencies not available. Install: chromadb, openai")

        import chromadb
        import openai

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        # OpenAI 클라이언트 초기화
        self.openai_client = openai.OpenAI(api_key=api_key)
        self.embedding_model_name = os.getenv(
            "OPENAI_EMBEDDING_MODEL",
            self.embedding_model_name or "text-embedding-3-small"
        )

        # ChromaDB 초기화 (modern API - persistent client)
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_dir)

        # 컬렉션 생성/로드
        self.collection = self.client.get_or_create_collection(
            name="amore_docs",
            metadata={"hnsw:space": "cosine"}
        )

        # 문서 인덱싱 (컬렉션이 비어있을 때만)
        if self.collection.count() == 0:
            await self._index_documents()

        print(f"ChromaDB initialized: {self.collection.count()} documents indexed")

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """OpenAI Embeddings API로 텍스트 임베딩 생성"""
        if not self.openai_client:
            return []
        response = self.openai_client.embeddings.create(
            model=self.embedding_model_name,
            input=texts
        )
        return [item.embedding for item in response.data]

    async def expand_query(self, query: str) -> List[str]:
        """
        LLM을 사용하여 쿼리 확장

        Args:
            query: 원본 쿼리

        Returns:
            확장된 쿼리 리스트 (원본 포함)
        """
        if not self.openai_client or not self.use_query_expansion:
            return [query]

        try:
            import openai

            prompt = f"""You are a search query expansion assistant. Given a user query, generate 2-3 alternative phrasings or related queries that would help retrieve relevant documents.

Original query: {query}

Generate alternative queries that:
1. Use synonyms or related terms
2. Rephrase the question differently
3. Add relevant context

Return ONLY a JSON array of strings, like: ["query1", "query2", "query3"]
Do not include any explanation."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are a search query expansion system."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )

            content = response.choices[0].message.content.strip()

            # JSON 파싱
            import json
            expanded = json.loads(content)

            # 원본 쿼리를 리스트 맨 앞에 추가
            if isinstance(expanded, list):
                return [query] + expanded[:3]  # 최대 3개까지
            else:
                return [query]

        except Exception as e:
            print(f"Query expansion failed: {e}")
            return [query]

    async def _index_documents(self) -> None:
        """문서 벡터 인덱싱"""
        if not self.collection or not self.openai_client:
            return

        ids = []
        documents = []
        metadatas = []

        for chunk in self.chunks:
            ids.append(chunk["id"])
            documents.append(chunk["content"])
            metadatas.append({
                "doc_id": chunk["doc_id"],
                "doc_type": chunk.get("doc_type", "metric_guide"),
                "title": chunk["title"],
                "description": chunk["description"],
                "content_type": chunk.get("content_type", "text"),
                "source_filename": chunk.get("source_filename", "")
            })

        if documents:
            batch_size = 100
            total_batches = (len(documents) - 1) // batch_size + 1
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]

                try:
                    embeddings = self._embed_texts(batch_docs)
                    if embeddings:
                        self.collection.add(
                            ids=batch_ids,
                            documents=batch_docs,
                            embeddings=embeddings,
                            metadatas=batch_metadatas
                        )
                        print(f"Indexed batch {i // batch_size + 1}/{total_batches}")
                except Exception as e:
                    print(f"Indexing batch failed: {e}")

    def _get_cache_key(
        self, 
        query: str, 
        top_k: int, 
        doc_filter: Optional[str],
        doc_type_filter: Optional[List[str]] = None
    ) -> str:
        """캐시 키 생성"""
        type_key = ",".join(doc_type_filter) if doc_type_filter else "all_types"
        return f"{query}:{top_k}:{doc_filter or 'all'}:{type_key}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """캐시 유효성 확인 (TTL 체크)"""
        if cache_key not in self._cache_timestamps:
            return False
        elapsed = time.time() - self._cache_timestamps[cache_key]
        return elapsed < self.get_cache_ttl()

    def _clean_expired_cache(self) -> None:
        """만료된 캐시 항목 정리"""
        current_time = time.time()
        cache_ttl = self.get_cache_ttl()
        expired_keys = [
            key for key, timestamp in self._cache_timestamps.items()
            if current_time - timestamp >= cache_ttl
        ]
        for key in expired_keys:
            self._search_cache.pop(key, None)
            self._cache_timestamps.pop(key, None)

    async def search(
        self,
        query: str,
        top_k: int = 3,
        doc_filter: Optional[str] = None,
        doc_type_filter: Optional[List[str]] = None,
        use_query_expansion: Optional[bool] = None,
        use_reranking: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        쿼리 기반 문서 검색 (TTL 캐싱 적용)

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            doc_filter: 특정 문서 ID로 필터링
            doc_type_filter: 문서 유형 필터링 (예: ["playbook", "intelligence"])
            use_query_expansion: Query Expansion 사용 여부 (None이면 생성자 설정 따름)
            use_reranking: Reranking 사용 여부 (None이면 생성자 설정 따름)

        Returns:
            검색 결과 리스트
        """
        if not self._initialized:
            await self.initialize()

        # 파라미터 기본값 설정
        if use_query_expansion is None:
            use_query_expansion = self.use_query_expansion
        if use_reranking is None:
            use_reranking = self.use_reranker

        # 캐시 키 생성 및 확인
        cache_key = self._get_cache_key(query, top_k, doc_filter, doc_type_filter)

        if self._is_cache_valid(cache_key):
            return self._search_cache[cache_key]

        # 만료된 캐시 정리 (주기적으로)
        if len(self._cache_timestamps) > 50:
            self._clean_expired_cache()

        # Query Expansion (옵션)
        queries = [query]
        if use_query_expansion:
            queries = await self.expand_query(query)

        # 각 쿼리에 대해 검색 수행
        all_results = []
        seen_ids = set()

        for q in queries:
            # 벡터 검색 (필수)
            if self.collection is not None and self.openai_client is not None:
                results = await self._vector_search(
                    q,
                    top_k * 3 if use_reranking else top_k,  # reranking 시 더 많은 후보 검색
                    doc_filter,
                    doc_type_filter
                )
            else:
                raise RuntimeError("Vector search is required but not available")

            # 중복 제거
            for result in results:
                if result["id"] not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(result["id"])

        # Reranking (옵션)
        if use_reranking and all_results:
            if self._reranker is None and RERANKER_AVAILABLE:
                self._reranker = get_reranker()

            if self._reranker:
                # Reranker 호출
                ranked = self._reranker.rerank(query, all_results, top_k=top_k)
                result = [
                    {
                        "id": doc.metadata.get("chunk_id", ""),
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "score": doc.score
                    }
                    for doc in ranked
                ]
            else:
                result = all_results[:top_k]
        else:
            result = all_results[:top_k]

        # 결과 캐싱
        self._search_cache[cache_key] = result
        self._cache_timestamps[cache_key] = time.time()

        return result

    async def _vector_search(
        self,
        query: str,
        top_k: int,
        doc_filter: Optional[str],
        doc_type_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """벡터 유사도 검색"""
        if not self.openai_client:
            return []
        query_embedding = self._embed_texts([query])
        if not query_embedding:
            return []

        # 필터 조건 구성
        where_filter = None
        if doc_filter and doc_type_filter:
            # doc_id와 doc_type 모두 필터링
            where_filter = {
                "$and": [
                    {"doc_id": doc_filter},
                    {"doc_type": {"$in": doc_type_filter}}
                ]
            }
        elif doc_filter:
            where_filter = {"doc_id": doc_filter}
        elif doc_type_filter:
            where_filter = {"doc_type": {"$in": doc_type_filter}}

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        search_results = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                chunk_id = results["ids"][0][i]
                chunk = self._chunk_index.get(chunk_id)
                if chunk:
                    metadata = {
                        "doc_id": chunk["doc_id"],
                        "doc_type": chunk.get("doc_type", "metric_guide"),
                        "title": chunk.get("title", ""),
                        "description": chunk.get("description", ""),
                        "keywords": chunk.get("keywords", []),
                        "content_type": chunk.get("content_type", "text"),
                        "chunk_id": chunk_id,
                        "source_filename": chunk.get("source_filename", ""),
                        "target_brand": chunk.get("target_brand"),
                        "brands_covered": chunk.get("brands_covered", [])
                    }
                else:
                    metadata = results["metadatas"][0][i]

                search_results.append({
                    "id": chunk_id,
                    "content": results["documents"][0][i],
                    "metadata": metadata,
                    "score": 1 - results["distances"][0][i]  # 거리를 유사도로 변환
                })

        return search_results

    async def _keyword_search(
        self,
        query: str,
        top_k: int,
        doc_filter: Optional[str],
        doc_type_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """키워드 기반 검색 (폴백)"""
        query_lower = query.lower()
        scored_chunks = []

        for chunk in self.chunks:
            # doc_id 필터
            if doc_filter and chunk["doc_id"] != doc_filter:
                continue
            
            # doc_type 필터
            if doc_type_filter and chunk.get("doc_type") not in doc_type_filter:
                continue

            score = 0

            # 키워드 매칭
            for keyword in chunk["keywords"]:
                if keyword.lower() in query_lower:
                    score += 2

            # 내용 매칭
            content_lower = chunk["content"].lower()
            query_words = query_lower.split()
            for word in query_words:
                if len(word) > 2 and word in content_lower:
                    score += 1

            if score > 0:
                scored_chunks.append({
                    "id": chunk["id"],
                    "content": chunk["content"],
                    "metadata": {
                        "doc_id": chunk["doc_id"],
                        "doc_type": chunk.get("doc_type", "metric_guide"),
                        "title": chunk["title"],
                        "description": chunk["description"],
                        "keywords": chunk.get("keywords", []),
                        "content_type": chunk.get("content_type", "text"),
                        "chunk_id": chunk["id"],
                        "source_filename": chunk.get("source_filename", ""),
                        "target_brand": chunk.get("target_brand"),
                        "brands_covered": chunk.get("brands_covered", [])
                    },
                    "score": score
                })

        # 점수순 정렬
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)

        return scored_chunks[:top_k]

    async def get_document(self, doc_id: str) -> Optional[str]:
        """특정 문서 전체 반환"""
        if not self._initialized:
            await self.initialize()

        return self.documents.get(doc_id)

    async def get_relevant_context(
        self,
        query: str,
        max_tokens: int = 2000
    ) -> str:
        """
        쿼리에 관련된 컨텍스트 반환 (LLM 프롬프트용)

        Args:
            query: 사용자 쿼리
            max_tokens: 최대 토큰 수 (대략적)

        Returns:
            관련 컨텍스트 문자열
        """
        results = await self.search(query, top_k=5)

        context_parts = []
        total_length = 0

        for result in results:
            content = result["content"]
            metadata = result["metadata"]

            # 토큰 제한 체크 (대략 4자 = 1토큰)
            if total_length + len(content) > max_tokens * 4:
                break

            context_parts.append(f"[{metadata.get('title', 'Unknown')}]\n{content}")
            total_length += len(content)

        return "\n\n---\n\n".join(context_parts)
