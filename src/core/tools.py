"""
에이전트 도구 정의
==================
OpenAI Function Calling 형식의 도구 정의

역할:
- LLM이 호출할 수 있는 도구(함수) 정의
- 도구 파라미터 스키마 정의
- 도구 실행 인터페이스 제공

연결 파일:
- agents/crawler_agent.py: crawl_amazon 도구
- agents/metrics_agent.py: calculate_metrics 도구
- core/models.py: ToolResult
- core/llm_orchestrator.py: 도구 실행 호출
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .models import ToolResult

logger = logging.getLogger(__name__)


# =============================================================================
# 도구 정의 타입
# =============================================================================


@dataclass
class ToolParameter:
    """도구 파라미터 정의"""

    name: str
    type: str  # string, integer, number, boolean, array, object
    description: str
    required: bool = False
    enum: list[str] | None = None
    default: Any = None


@dataclass
class AgentTool:
    """
    에이전트 도구 정의

    OpenAI Function Calling 형식으로 변환 가능.

    Attributes:
        name: 도구 이름 (함수명)
        description: 도구 설명
        parameters: 파라미터 목록
        executor: 실행 함수 (async)
        requires_data: 데이터 필요 여부
        requires_crawl: 크롤링 결과 필요 여부
    """

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    executor: Callable[..., Awaitable[dict[str, Any]]] | None = None
    requires_data: bool = False
    requires_crawl: bool = False

    def to_openai_schema(self) -> dict[str, Any]:
        """
        OpenAI Function Calling 스키마로 변환

        Returns:
            OpenAI 호환 함수 스키마
        """
        properties = {}
        required = []

        for param in self.parameters:
            prop = {"type": param.type, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        }


# =============================================================================
# 표준 도구 정의
# =============================================================================

# 1. 크롤링 도구
CRAWL_AMAZON_TOOL = AgentTool(
    name="crawl_amazon",
    description="Amazon에서 최신 제품 순위 데이터를 크롤링합니다. 하루에 한 번만 실행하면 됩니다.",
    parameters=[
        ToolParameter(
            name="categories",
            type="array",
            description="크롤링할 카테고리 목록 (비우면 전체)",
            required=False,
        ),
        ToolParameter(
            name="max_products",
            type="integer",
            description="카테고리당 최대 제품 수",
            required=False,
            default=100,
        ),
    ],
    requires_data=False,
    requires_crawl=False,
)

# 2. 지표 계산 도구
CALCULATE_METRICS_TOOL = AgentTool(
    name="calculate_metrics",
    description="크롤링된 데이터를 기반으로 SoS, HHI, CPI 등 지표를 계산합니다.",
    parameters=[
        ToolParameter(
            name="category_id",
            type="string",
            description="지표 계산할 카테고리 ID (비우면 전체)",
            required=False,
        ),
        ToolParameter(
            name="include_historical",
            type="boolean",
            description="히스토리 데이터 포함 여부",
            required=False,
            default=True,
        ),
    ],
    requires_data=True,
    requires_crawl=True,
)

# 3. 데이터 조회 도구
QUERY_DATA_TOOL = AgentTool(
    name="query_data",
    description="저장된 데이터에서 특정 브랜드/제품/카테고리 정보를 조회합니다.",
    parameters=[
        ToolParameter(
            name="query_type",
            type="string",
            description="조회 유형",
            required=True,
            enum=["brand_metrics", "product_rank", "category_summary", "competitor_analysis"],
        ),
        ToolParameter(name="brand", type="string", description="조회할 브랜드명", required=False),
        ToolParameter(
            name="category_id", type="string", description="조회할 카테고리 ID", required=False
        ),
        ToolParameter(
            name="time_range",
            type="string",
            description="조회 기간",
            required=False,
            enum=["today", "7days", "30days", "90days"],
        ),
    ],
    requires_data=True,
    requires_crawl=False,
)

# 4. KG 조회 도구
QUERY_KG_TOOL = AgentTool(
    name="query_knowledge_graph",
    description="지식 그래프에서 브랜드 관계, 경쟁사, 제품 정보를 조회합니다.",
    parameters=[
        ToolParameter(
            name="entity",
            type="string",
            description="조회할 엔티티 (브랜드명 또는 ASIN)",
            required=True,
        ),
        ToolParameter(
            name="relation_type",
            type="string",
            description="조회할 관계 유형",
            required=False,
            enum=["competitors", "products", "category", "all"],
        ),
        ToolParameter(
            name="depth", type="integer", description="탐색 깊이", required=False, default=1
        ),
    ],
    requires_data=False,
    requires_crawl=False,
)

# 5. 직접 응답 (도구 호출 없이 응답)
DIRECT_ANSWER_TOOL = AgentTool(
    name="direct_answer",
    description="컨텍스트만으로 충분히 답변할 수 있을 때 사용합니다. 추가 도구 호출이 필요 없습니다.",
    parameters=[
        ToolParameter(name="reason", type="string", description="직접 응답하는 이유", required=True)
    ],
    requires_data=False,
    requires_crawl=False,
)

# 6. 경쟁사 Deals 조회 도구
QUERY_DEALS_TOOL = AgentTool(
    name="query_deals",
    description="경쟁사 할인 정보(Lightning Deal, 쿠폰, 할인율)를 조회합니다. 경쟁사 프로모션 상황을 파악할 때 사용합니다.",
    parameters=[
        ToolParameter(
            name="brand",
            type="string",
            description="조회할 브랜드명 (비우면 전체 경쟁사)",
            required=False,
        ),
        ToolParameter(
            name="hours",
            type="integer",
            description="최근 N시간 데이터 조회 (기본: 24시간)",
            required=False,
            default=24,
        ),
        ToolParameter(
            name="deal_type",
            type="string",
            description="딜 유형 필터",
            required=False,
            enum=["lightning", "deal_of_day", "best_deal", "coupon", "all"],
        ),
        ToolParameter(
            name="min_discount",
            type="integer",
            description="최소 할인율 필터 (예: 20 = 20% 이상)",
            required=False,
        ),
    ],
    requires_data=False,
    requires_crawl=False,
)

# 7. Deals 요약 통계 도구
QUERY_DEALS_SUMMARY_TOOL = AgentTool(
    name="query_deals_summary",
    description="경쟁사 할인 현황 요약 통계를 조회합니다. 브랜드별 딜 현황, 일별 추이, 평균 할인율 등을 확인합니다.",
    parameters=[
        ToolParameter(
            name="days",
            type="integer",
            description="분석 기간 (일 단위, 기본: 7일)",
            required=False,
            default=7,
        ),
        ToolParameter(
            name="group_by",
            type="string",
            description="그룹화 기준",
            required=False,
            enum=["brand", "date", "deal_type"],
        ),
    ],
    requires_data=False,
    requires_crawl=False,
)


# 8. 브랜드 상태 조회 도구 (v3 포팅)
GET_BRAND_STATUS_TOOL = AgentTool(
    name="get_brand_status",
    description="LANEIGE 브랜드의 현재 KPI 상태를 조회합니다. SoS, 순위, Top10 제품 수 등을 확인합니다.",
    parameters=[],
    requires_data=True,
    requires_crawl=False,
)

# 9. 제품 정보 조회 도구 (v3 포팅)
GET_PRODUCT_INFO_TOOL = AgentTool(
    name="get_product_info",
    description="특정 LANEIGE 제품의 상세 정보를 조회합니다. ASIN 또는 제품명으로 검색합니다.",
    parameters=[
        ToolParameter(
            name="product_name",
            type="string",
            description="제품명 또는 ASIN (예: 'Lip Sleeping Mask' 또는 'B09...')",
            required=True,
        )
    ],
    requires_data=True,
    requires_crawl=False,
)

# 10. 경쟁사 분석 도구 (v3 포팅)
GET_COMPETITOR_ANALYSIS_TOOL = AgentTool(
    name="get_competitor_analysis",
    description="경쟁사 대비 LANEIGE의 포지션을 분석합니다. 특정 브랜드 또는 전체 경쟁사 현황을 조회합니다.",
    parameters=[
        ToolParameter(
            name="brand_name",
            type="string",
            description="분석할 경쟁 브랜드명 (비우면 전체 경쟁사)",
            required=False,
        )
    ],
    requires_data=True,
    requires_crawl=False,
)

# 11. 카테고리 정보 도구 (v3 포팅)
GET_CATEGORY_INFO_TOOL = AgentTool(
    name="get_category_info",
    description="카테고리별 KPI와 트렌드 정보를 조회합니다.",
    parameters=[
        ToolParameter(
            name="category",
            type="string",
            description="카테고리명 (예: 'lip_care', 'skin_care'). 비우면 전체 카테고리.",
            required=False,
        )
    ],
    requires_data=True,
    requires_crawl=False,
)

# 12. 액션 아이템 도구 (v3 포팅)
GET_ACTION_ITEMS_TOOL = AgentTool(
    name="get_action_items",
    description="현재 LANEIGE 브랜드의 액션 아이템 및 상태 요약을 조회합니다.",
    parameters=[],
    requires_data=True,
    requires_crawl=False,
)


# =============================================================================
# 도구 레지스트리
# =============================================================================

# 기본 도구 목록
AGENT_TOOLS: dict[str, AgentTool] = {
    "crawl_amazon": CRAWL_AMAZON_TOOL,
    "calculate_metrics": CALCULATE_METRICS_TOOL,
    "query_data": QUERY_DATA_TOOL,
    "query_knowledge_graph": QUERY_KG_TOOL,
    "direct_answer": DIRECT_ANSWER_TOOL,
    "query_deals": QUERY_DEALS_TOOL,
    "query_deals_summary": QUERY_DEALS_SUMMARY_TOOL,
    # v3 대시보드 도구 (포팅)
    "get_brand_status": GET_BRAND_STATUS_TOOL,
    "get_product_info": GET_PRODUCT_INFO_TOOL,
    "get_competitor_analysis": GET_COMPETITOR_ANALYSIS_TOOL,
    "get_category_info": GET_CATEGORY_INFO_TOOL,
    "get_action_items": GET_ACTION_ITEMS_TOOL,
}


def get_all_tool_schemas() -> list[dict[str, Any]]:
    """모든 도구의 OpenAI 스키마 반환"""
    return [tool.to_openai_schema() for tool in AGENT_TOOLS.values()]


def get_tool(name: str) -> AgentTool | None:
    """이름으로 도구 조회"""
    return AGENT_TOOLS.get(name)


def get_tools_for_context(has_data: bool, has_crawl: bool) -> list[dict[str, Any]]:
    """
    현재 상황에 맞는 도구만 반환

    Args:
        has_data: 데이터 있음
        has_crawl: 크롤링 완료

    Returns:
        사용 가능한 도구 스키마 목록
    """
    available = []

    for tool in AGENT_TOOLS.values():
        # 데이터 필요한 도구는 데이터 있을 때만
        if tool.requires_data and not has_data:
            continue
        # 크롤링 필요한 도구는 크롤링 완료 시만
        if tool.requires_crawl and not has_crawl:
            continue
        available.append(tool.to_openai_schema())

    return available


# =============================================================================
# 도구 실행기
# =============================================================================


class ToolExecutor:
    """
    도구 실행기

    등록된 에이전트들을 실행하고 결과를 ToolResult로 반환.

    Usage:
        executor = ToolExecutor()
        executor.register_agent("crawl_amazon", crawler_agent)
        result = await executor.execute("crawl_amazon", {"categories": ["lip_care"]})
    """

    def __init__(self):
        """초기화"""
        self._agents: dict[str, Any] = {}
        self._executors: dict[str, Callable] = {}

    def register_agent(self, tool_name: str, agent: Any) -> None:
        """
        도구에 에이전트 연결

        Args:
            tool_name: 도구 이름
            agent: 에이전트 인스턴스 (execute 메서드 필요)
        """
        self._agents[tool_name] = agent
        logger.debug(f"Registered agent for tool: {tool_name}")

    def register_executor(
        self, tool_name: str, executor: Callable[..., Awaitable[dict[str, Any]]]
    ) -> None:
        """
        도구에 실행 함수 연결

        Args:
            tool_name: 도구 이름
            executor: 비동기 실행 함수
        """
        self._executors[tool_name] = executor
        logger.debug(f"Registered executor for tool: {tool_name}")

    async def execute(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        """
        도구 실행

        Args:
            tool_name: 실행할 도구 이름
            params: 도구 파라미터

        Returns:
            ToolResult
        """
        start_time = datetime.now()

        # direct_answer는 실행 없이 성공 반환
        if tool_name == "direct_answer":
            return ToolResult(
                tool_name=tool_name,
                success=True,
                data={"reason": params.get("reason", "컨텍스트로 응답")},
                execution_time_ms=0,
            )

        try:
            # 에이전트 확인
            if tool_name in self._agents:
                agent = self._agents[tool_name]
                result = await agent.execute(**params)

            # 실행 함수 확인
            elif tool_name in self._executors:
                executor = self._executors[tool_name]
                result = await executor(**params)

            else:
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    error=f"도구 '{tool_name}'에 연결된 실행기가 없습니다",
                    execution_time_ms=0,
                )

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return ToolResult(
                tool_name=tool_name,
                success=True,
                data=result if isinstance(result, dict) else {"result": result},
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Tool execution failed: {tool_name} - {e}")

            return ToolResult(
                tool_name=tool_name, success=False, error=str(e), execution_time_ms=execution_time
            )

    def get_available_tools(self) -> list[str]:
        """사용 가능한 도구 목록"""
        available = set(self._agents.keys()) | set(self._executors.keys())
        available.add("direct_answer")  # 항상 사용 가능
        return list(available)

    def is_tool_available(self, tool_name: str) -> bool:
        """도구 사용 가능 여부"""
        if tool_name == "direct_answer":
            return True
        return tool_name in self._agents or tool_name in self._executors
