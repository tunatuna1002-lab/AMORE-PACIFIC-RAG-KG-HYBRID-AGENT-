# 정석적인 RAG + 지식그래프 + 온톨로지 에이전트 구조 (조사 정리, 2026-09-02)

> 목적: 이 프로젝트의 온톨로지 실효화 트랙(F9)과 검색 스택 재설계(F3)의 기준점. 각 주장 뒤에 출처를 붙였다. 수치는 해당 논문의 보고값이며 이 프로젝트에서 재현한 값이 아니다.

## 1. 한 줄 정의

> 정석 구조 = 문서 층·지식그래프 층·온톨로지(의미) 층을 **상호 색인**하고, 에이전트가 질문을 **논리형(logical form)** 으로 분해해 층별 연산자(벡터 검색·그래프 탐색·규칙 추론·수치 계산)를 선택·검증하는 구조.

근거: GraphRAG 서베이는 워크플로를 그래프 기반 색인 → 그래프 유도 검색 → 그래프 강화 생성 3단계로 형식화한다(arXiv 2408.08921). VLDB 통합 프레임워크는 12개 방법을 그래프 구축 → 인덱스 구축 → 연산자 구성 → 검색·생성 4단계로 환원한다(arXiv 2503.04338). KAG는 KG↔원문 청크 상호 색인과 논리형 유도 하이브리드 추론(검색·KG 추론·언어 추론·수치 계산 4연산자)을 핵심으로 둔다(arXiv 2409.13731, OpenSPG/KAG).

## 2. 언제 그래프가 이득인가

| 질문 유형 | 유리한 쪽 | 근거 |
|-----------|-----------|------|
| 단일 홉 사실 조회("SoS 정의가 뭐야") | 벡터 RAG로 충분 | GraphRAG-Bench(ICLR'26): 단순 사실 검색은 격차 미미 |
| 다중 홉("LANEIGE 모기업의 다른 브랜드가 Lip Care에서 어떤 위치") | 그래프 | 같은 벤치: 복합 추론·맥락 요약에서 일관 우위, 단 지연 약 2.3배 |
| 코퍼스 전체 주제 요약 | 그래프(커뮤니티 요약) | Microsoft GraphRAG global search(arXiv 2404.16130) |
| 에이전트형 반복 검색 | 격차 축소, 다중 홉은 여전히 그래프 안정 | "Do We Still Need GraphRAG?"(arXiv 2604.09666) |

결론: 그래프는 "있으면 좋은 것"이 아니라 **다중 홉·설명 가능 경로·관계형 데이터**일 때 쓰는 도구다. 이 프로젝트의 데이터(브랜드–제품–카테고리–순위–그룹)는 관계형이므로 조건을 만족하지만, 지표 정의 질문까지 그래프로 보낼 이유는 없다.

## 3. 층별 정석 설계

### 3.1 온톨로지 층 (T-Box)

- **방법론**: LOT·NeOn·METHONTOLOGY 공통 골격은 요구사항 명세(역량 질문, competency question) → 재사용 → 구현(OWL 2) → 평가(CQ→SPARQL 검증)이다. CQ는 온톨로지의 범위를 규정하고 동시에 테스트가 된다(NeOn; "Use of Competency Questions in Ontology Engineering: A Survey").
- **표현 선택**: 추론·검증이 필요하면 RDF/OWL + SHACL, 탐색·성능이 우선이면 라벨 속성 그래프(LPG). 형식 의미론은 RDF/OWL에만 있다(TigerGraph·Enterprise Knowledge 비교 글). 이 프로젝트는 owlready2 + JSON 트리플 저장소이므로 사실상 RDF 편에 있고, 규칙 추론을 쓰므로 유지가 합리적.
- **검증 두 겹**: T-Box 일관성(reasoner consistency) + A-Box 형태 검증(SHACL shapes). SHACL 위반 보고서는 표준 형식이라 LLM 출력 검증에도 재사용된다(arXiv 2604.00555, 2506.10678).
- **스키마 크기가 성능을 좌우**: 스키마 크기·표현력·표기 형식이 에이전트의 KG 질의 성공률에 직접 영향을 준다(arXiv 2507.09389). 소비자 없는 공리를 늘리면 오히려 해롭다.

### 3.2 지식그래프 구축 층 (A-Box)

- **3단 파이프라인**: 온톨로지 공학 → 지식 추출 → 지식 융합(arXiv 2510.20345). 스키마 기반 방식은 구조·정규화·일관성을, 스키마 자유 방식은 유연성을 준다.
- **이 프로젝트에 맞는 방식**: 원천이 이미 구조화(크롤 레코드·지표)이므로 추출은 LLM이 아니라 **매핑**이다(R2RML류 결정적 변환). LLM 추출은 비정형 원천(IR 보고서·뉴스)에만 스키마 가이드로 쓴다.
- **지식 융합 = 엔티티 해소**: 동일 개체 병합·중복 제거·정규화(KGGEN·LLM-Align 등). 이 프로젝트의 `LANEIGE`/`laneige` 이중 개체 문제가 정확히 이 단계 부재다.
- **출처(provenance)**: 트리플마다 원천(스냅샷 날짜·규칙 id)을 남긴다. 설명성과 L4 평가의 전제.

### 3.3 문서·검색 층

- **하이브리드 검색**: 밀집 + 희소(BM25) + RRF 융합이 기본(Neo4j GraphRAG 패키지의 HybridRetriever·HybridCypherRetriever). 이 프로젝트는 이미 갖고 있으나 RRF 구현이 3개다.
- **상호 색인**: 청크 → 언급 개체, 개체 → 청크. 검색 결과를 개체로 확장하고 개체에서 청크로 되돌아오는 경로가 다중 홉의 실체(KAG mutual indexing; Neo4j VectorCypherRetriever).
- **온톨로지 접지 검색(OG-RAG)**: 문서를 온톨로지로 접지된 사실 묶음(hyperedge)으로 조직하고 질의에 대해 최소 덮개 집합(set cover)을 검색. 사실 recall +55%, 정답률 +40% 보고(EMNLP 2025, arXiv 2412.15235). "브랜드×카테고리×날짜" 사실 묶음이 있는 이 프로젝트에 직접 적용 가능한 패턴.
- **계층 요약(GraphRAG community)**: 코퍼스 358청크 규모에서는 비용 대비 이득이 작다[추정]. 후순위.

### 3.4 의미 층 (semantic layer: 지표 정의)

- **정의**: 지표(measure)·차원(dimension)·정책을 한 곳에 정의하고 에이전트는 raw 계산 대신 정의된 지표를 호출한다.
- **효과**: 의미 층 유무를 짝지어 비교한 벤치에서 정확도 +17~23%p, 환각률 약 63% → 1.7% 보고(arXiv 2604.25149, Cube 블로그; 벤더 관여 자료이므로 수치는 확인 필요).
- **이 프로젝트 적용**: SoS/HHI/CPI가 15곳 이상에서 재계산되는 문제의 정석 해법. `MetricCalculator`를 의미 층으로 승격하고 API·규칙·인사이트가 모두 그것만 호출.

### 3.5 에이전트 층

- **패턴**: 계획(planning)·도구 사용·반성(reflection)·다중 에이전트(Agentic RAG 서베이 arXiv 2501.09136). "언제·무엇을·어떻게 검색할지"를 모델이 결정.
- **논리형 유도**: 질문 → 서브질의 DAG → 연산자 선택(벡터 / 그래프 탐색 / SPARQL·Cypher / 지표 계산 / 규칙 추론) → 검증 → 출처 포함 답변(KAG).
- **온톨로지가 에이전트를 제약**: 비대칭 결합(asymmetric coupling) — 온톨로지가 컨텍스트 조립·도구 발견·거버넌스 임계값 등 **입력**을 제약하고, SHACL 등으로 **출력**을 검증. 5개 산업 1,800회 실험에서 접지 에이전트가 유의하게 우위 보고(arXiv 2604.00555).
- **Text2Cypher/SPARQL**: 질문 변형에 가장 일관된 검색 방식으로 보고(Neo4j). 단 스키마 표현에 민감(3.1 참조).

## 4. 평가 (층별)

| 층 | 지표 | 이 프로젝트 현황 |
|----|------|------------------|
| 검색(L2) | Recall@k, nDCG | 있음 |
| KG(L3) | 개체·엣지 recall | 골든셋 필드 존재, 산출 미연결 |
| 추론(L4) | 추론 사실의 provenance 적중 | 없음 |
| 답변(L5) | faithfulness·attribution (RAGAS, NLI 기반 주장 분해) | 부분 |
| 온톨로지 | CQ→질의 통과율 | 없음 |
| 유형별 | 사실/추론/요약 분리 채점(GraphRAG-Bench 방식) | 없음 |

## 5. 이 프로젝트 매핑

| 정석 요소 | 현재 | 격차 |
|-----------|------|------|
| CQ 기반 T-Box | 제한 클래스 3개, CQ 없음 | CQ 12개부터 |
| 엔티티 해소 | 없음(대소문자 이중 개체) | Builder 단일 정규화 |
| A-Box 검증(SHACL/일관성) | 예외 삼킴 | 배치 실패로 승격 |
| 상호 색인 | 청크→개체 없음 | EntityLinker 결과를 청크 메타에 기록 |
| 온톨로지 접지 검색 | 미도달(D4) | OG-RAG식 사실 묶음 |
| 의미 층 | 15곳 재계산 | MetricCalculator 단일화 |
| 논리형 에이전트 | 키워드 의도 4벌 | 서브질의 DAG + 연산자 선택 |
| 오프라인 추론·온라인 소비 | 요청 시 OWL 생성 시도 | 배치 물질화 |
| 층별 평가 | L2만 실효 | L3·L4·CQ 게이트 |

## 6. 출처

- Graph RAG 서베이: https://arxiv.org/abs/2408.08921 , https://arxiv.org/abs/2501.00309 , https://arxiv.org/abs/2501.13958 , https://github.com/DEEP-PolyU/Awesome-GraphRAG
- 통합 프레임워크(VLDB 2025): https://arxiv.org/abs/2503.04338
- GraphRAG-Bench(ICLR'26): https://arxiv.org/abs/2506.05690 , https://github.com/GraphRAG-Bench/GraphRAG-Benchmark
- Do We Still Need GraphRAG?: https://arxiv.org/abs/2604.09666
- Microsoft GraphRAG: https://arxiv.org/pdf/2404.16130 , https://microsoft.github.io/graphrag/
- OG-RAG(EMNLP 2025): https://arxiv.org/abs/2412.15235 , https://aclanthology.org/2025.emnlp-main.1674/ , https://github.com/microsoft/ograg2
- KAG: https://arxiv.org/abs/2409.13731 , https://github.com/openspg/kag
- 온톨로지 공학·CQ: https://www.researchgate.net/publication/272829912_The_NeOn_Methodology_for_Ontology_Engineering , https://dl.acm.org/doi/10.1007/978-3-031-47262-6_3 , https://dgarijo.com/papers/iswc_llm.pdf
- 지식 개념화와 RAG 효능: https://arxiv.org/abs/2507.09389
- LLM 기반 KG 구축 서베이: https://arxiv.org/abs/2510.20345
- 온톨로지 제약 신경기호 에이전트: https://arxiv.org/abs/2604.00555
- 의미 층 벤치: https://arxiv.org/abs/2604.25149 , https://cube.dev/blog/why-semantic-layers-make-llm-analytics-reliable-a-paired-benchmark-across-three-frontier-models
- RDF vs LPG: https://www.tigergraph.com/blog/rdf-vs-property-graph-choosing-the-right-foundation-for-knowledge-graphs/ , https://enterprise-knowledge.com/cutting-through-the-noise-an-introduction-to-rdf-lpg-graphs/
- Neo4j GraphRAG 패턴: https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_rag.html , https://graphacademy.neo4j.com/courses/genai-workshop-graphrag/2-neo4j-graphrag/5-hybrid-cypher-retriever/
- Agentic RAG 서베이: https://arxiv.org/abs/2501.09136
- RAG 평가: https://www.getmaxim.ai/articles/complete-guide-to-rag-evaluation-metrics-methods-and-best-practices-for-2025/ , https://arxiv.org/pdf/2509.03626
