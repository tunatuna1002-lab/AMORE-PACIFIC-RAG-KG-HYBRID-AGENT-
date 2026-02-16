"""
Tools (AgentTool, ToolExecutor) 단위 테스트
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.tools import (
    AGENT_TOOLS,
    CRAWL_AMAZON_TOOL,
    DIRECT_ANSWER_TOOL,
    AgentTool,
    ToolExecutor,
    ToolParameter,
    get_all_tool_schemas,
    get_tool,
    get_tools_for_context,
)

# =============================================================================
# ToolParameter 테스트
# =============================================================================


class TestToolParameter:
    """ToolParameter 데이터클래스 테스트"""

    def test_create_basic(self):
        """기본 파라미터 생성"""
        param = ToolParameter(
            name="category",
            type="string",
            description="카테고리명",
        )
        assert param.name == "category"
        assert param.type == "string"
        assert param.required is False
        assert param.enum is None
        assert param.default is None

    def test_create_with_enum(self):
        """enum 포함 파라미터"""
        param = ToolParameter(
            name="query_type",
            type="string",
            description="조회 유형",
            required=True,
            enum=["brand", "product"],
        )
        assert param.required is True
        assert param.enum == ["brand", "product"]

    def test_create_with_default(self):
        """기본값 포함 파라미터"""
        param = ToolParameter(
            name="max_products",
            type="integer",
            description="최대 수",
            default=100,
        )
        assert param.default == 100


# =============================================================================
# AgentTool 테스트
# =============================================================================


class TestAgentTool:
    """AgentTool 데이터클래스 테스트"""

    def test_create_basic(self):
        """기본 도구 생성"""
        tool = AgentTool(name="test_tool", description="Test tool")
        assert tool.name == "test_tool"
        assert tool.description == "Test tool"
        assert tool.parameters == []
        assert tool.executor is None
        assert tool.requires_data is False
        assert tool.requires_crawl is False

    def test_to_openai_schema_no_params(self):
        """파라미터 없는 도구 스키마"""
        tool = AgentTool(name="simple", description="Simple tool")
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "simple"
        assert schema["function"]["description"] == "Simple tool"
        assert schema["function"]["parameters"]["properties"] == {}
        assert schema["function"]["parameters"]["required"] == []

    def test_to_openai_schema_with_params(self):
        """파라미터 있는 도구 스키마"""
        tool = AgentTool(
            name="query",
            description="Query tool",
            parameters=[
                ToolParameter(
                    name="brand",
                    type="string",
                    description="브랜드명",
                    required=True,
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="최대 수",
                    required=False,
                    default=10,
                ),
            ],
        )
        schema = tool.to_openai_schema()
        props = schema["function"]["parameters"]["properties"]
        assert "brand" in props
        assert "limit" in props
        assert props["brand"]["type"] == "string"
        assert props["limit"]["default"] == 10
        assert "brand" in schema["function"]["parameters"]["required"]
        assert "limit" not in schema["function"]["parameters"]["required"]

    def test_to_openai_schema_with_enum(self):
        """enum 포함 파라미터 스키마"""
        tool = AgentTool(
            name="filter",
            description="Filter",
            parameters=[
                ToolParameter(
                    name="type",
                    type="string",
                    description="유형",
                    enum=["a", "b", "c"],
                ),
            ],
        )
        schema = tool.to_openai_schema()
        assert schema["function"]["parameters"]["properties"]["type"]["enum"] == ["a", "b", "c"]


# =============================================================================
# 표준 도구 정의 테스트
# =============================================================================


class TestStandardTools:
    """표준 도구 정의 테스트"""

    def test_crawl_amazon_tool_defined(self):
        """CRAWL_AMAZON_TOOL 정의 확인"""
        assert CRAWL_AMAZON_TOOL.name == "crawl_amazon"
        assert not CRAWL_AMAZON_TOOL.requires_data

    def test_direct_answer_tool_defined(self):
        """DIRECT_ANSWER_TOOL 정의 확인"""
        assert DIRECT_ANSWER_TOOL.name == "direct_answer"
        assert not DIRECT_ANSWER_TOOL.requires_data
        assert not DIRECT_ANSWER_TOOL.requires_crawl

    def test_agent_tools_registry_populated(self):
        """AGENT_TOOLS 레지스트리 비어있지 않음"""
        assert len(AGENT_TOOLS) >= 5
        assert "crawl_amazon" in AGENT_TOOLS
        assert "direct_answer" in AGENT_TOOLS
        assert "query_data" in AGENT_TOOLS


# =============================================================================
# 모듈 함수 테스트
# =============================================================================


class TestModuleFunctions:
    """모듈 레벨 함수 테스트"""

    def test_get_all_tool_schemas(self):
        """get_all_tool_schemas 반환"""
        schemas = get_all_tool_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) == len(AGENT_TOOLS)
        for schema in schemas:
            assert schema["type"] == "function"
            assert "function" in schema

    def test_get_tool_existing(self):
        """존재하는 도구 조회"""
        tool = get_tool("crawl_amazon")
        assert tool is not None
        assert tool.name == "crawl_amazon"

    def test_get_tool_nonexistent(self):
        """존재하지 않는 도구 조회"""
        tool = get_tool("nonexistent_tool")
        assert tool is None

    def test_get_tools_for_context_no_data_no_crawl(self):
        """데이터/크롤링 없을 때 사용 가능 도구"""
        tools = get_tools_for_context(has_data=False, has_crawl=False)
        names = [t["function"]["name"] for t in tools]
        assert "direct_answer" in names
        assert "crawl_amazon" in names
        # requires_data=True 도구는 제외
        for t in tools:
            tool_obj = get_tool(t["function"]["name"])
            assert not tool_obj.requires_data

    def test_get_tools_for_context_with_data_and_crawl(self):
        """데이터/크롤링 모두 있을 때 모든 도구 사용 가능"""
        tools = get_tools_for_context(has_data=True, has_crawl=True)
        assert len(tools) == len(AGENT_TOOLS)

    def test_get_tools_for_context_with_data_no_crawl(self):
        """데이터만 있을 때"""
        tools = get_tools_for_context(has_data=True, has_crawl=False)
        names = [t["function"]["name"] for t in tools]
        # requires_crawl=True 도구는 제외
        for name in names:
            tool_obj = get_tool(name)
            assert not tool_obj.requires_crawl


# =============================================================================
# ToolExecutor 테스트
# =============================================================================


class TestToolExecutorInit:
    """ToolExecutor 초기화 테스트"""

    def test_init(self):
        """초기화"""
        executor = ToolExecutor()
        assert executor._agents == {}
        assert executor._executors == {}


class TestToolExecutorRegister:
    """ToolExecutor 등록 테스트"""

    def test_register_agent(self):
        """에이전트 등록"""
        executor = ToolExecutor()
        mock_agent = MagicMock()
        executor.register_agent("crawl_amazon", mock_agent)
        assert "crawl_amazon" in executor._agents

    def test_register_executor(self):
        """실행 함수 등록"""
        executor = ToolExecutor()
        mock_fn = AsyncMock()
        executor.register_executor("custom_tool", mock_fn)
        assert "custom_tool" in executor._executors


class TestToolExecutorExecute:
    """ToolExecutor.execute 테스트"""

    @pytest.mark.asyncio
    async def test_execute_direct_answer(self):
        """direct_answer 도구 실행"""
        executor = ToolExecutor()
        result = await executor.execute("direct_answer", {"reason": "충분한 컨텍스트"})
        assert result.success is True
        assert result.tool_name == "direct_answer"
        assert result.data["reason"] == "충분한 컨텍스트"
        assert result.execution_time_ms == 0

    @pytest.mark.asyncio
    async def test_execute_direct_answer_no_reason(self):
        """direct_answer 이유 없이 실행"""
        executor = ToolExecutor()
        result = await executor.execute("direct_answer", {})
        assert result.success is True
        assert result.data["reason"] == "컨텍스트로 응답"

    @pytest.mark.asyncio
    async def test_execute_with_agent(self):
        """에이전트를 통한 도구 실행"""
        executor = ToolExecutor()
        mock_agent = MagicMock()
        mock_agent.execute = AsyncMock(return_value={"products": 100})
        executor.register_agent("crawl_amazon", mock_agent)

        result = await executor.execute("crawl_amazon", {"categories": ["lip_care"]})
        assert result.success is True
        assert result.data["products"] == 100
        mock_agent.execute.assert_called_once_with(categories=["lip_care"])

    @pytest.mark.asyncio
    async def test_execute_with_executor_function(self):
        """실행 함수를 통한 도구 실행"""
        executor = ToolExecutor()
        mock_fn = AsyncMock(return_value={"result": "done"})
        executor.register_executor("custom_tool", mock_fn)

        result = await executor.execute("custom_tool", {"param": "value"})
        assert result.success is True
        assert result.data["result"] == "done"
        mock_fn.assert_called_once_with(param="value")

    @pytest.mark.asyncio
    async def test_execute_unregistered_tool(self):
        """등록되지 않은 도구 실행"""
        executor = ToolExecutor()
        result = await executor.execute("unknown_tool", {})
        assert result.success is False
        assert "연결된 실행기가 없습니다" in result.error

    @pytest.mark.asyncio
    async def test_execute_agent_raises_exception(self):
        """에이전트 실행 중 예외"""
        executor = ToolExecutor()
        mock_agent = MagicMock()
        mock_agent.execute = AsyncMock(side_effect=RuntimeError("Agent failed"))
        executor.register_agent("failing_tool", mock_agent)

        result = await executor.execute("failing_tool", {})
        assert result.success is False
        assert "Agent failed" in result.error
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_execute_non_dict_result_wrapped(self):
        """비-dict 결과는 래핑됨"""
        executor = ToolExecutor()
        mock_agent = MagicMock()
        mock_agent.execute = AsyncMock(return_value="string_result")
        executor.register_agent("str_tool", mock_agent)

        result = await executor.execute("str_tool", {})
        assert result.success is True
        assert result.data == {"result": "string_result"}


class TestToolExecutorAvailability:
    """ToolExecutor 도구 가용성 테스트"""

    def test_get_available_tools_empty(self):
        """빈 상태에서 direct_answer만 사용 가능"""
        executor = ToolExecutor()
        tools = executor.get_available_tools()
        assert "direct_answer" in tools

    def test_get_available_tools_with_registered(self):
        """등록된 도구 포함"""
        executor = ToolExecutor()
        executor.register_agent("crawl", MagicMock())
        executor.register_executor("custom", AsyncMock())
        tools = executor.get_available_tools()
        assert "crawl" in tools
        assert "custom" in tools
        assert "direct_answer" in tools

    def test_is_tool_available_direct_answer(self):
        """direct_answer 항상 사용 가능"""
        executor = ToolExecutor()
        assert executor.is_tool_available("direct_answer") is True

    def test_is_tool_available_registered(self):
        """등록된 도구 사용 가능"""
        executor = ToolExecutor()
        executor.register_agent("crawl", MagicMock())
        assert executor.is_tool_available("crawl") is True

    def test_is_tool_available_unregistered(self):
        """미등록 도구 사용 불가"""
        executor = ToolExecutor()
        assert executor.is_tool_available("unknown") is False
