"""
Chat Workflow
=============
Orchestrates the chat query processing flow.

Flow:
1. Analyze query complexity and intent
2. Retrieve context (RAG + KG)
3. Generate response via chatbot
4. Return structured result

Clean Architecture:
- Depends only on domain interfaces (ChatbotAgentProtocol, RetrieverProtocol)
- No infrastructure dependencies
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.application.services.query_analyzer import QueryAnalyzer
from src.domain.interfaces.chatbot import ChatbotAgentProtocol
from src.domain.interfaces.retriever import RetrieverProtocol


@dataclass
class ChatWorkflowResult:
    """Chat workflow execution result"""

    query: str
    response: str | None = None
    complexity: str = "simple"
    intent: str = "general"
    sources: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    suggestions: list[str] = field(default_factory=list)
    entities: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    execution_time: float = 0.0
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "query": self.query,
            "response": self.response,
            "complexity": self.complexity,
            "intent": self.intent,
            "sources": self.sources,
            "confidence": self.confidence,
            "suggestions": self.suggestions,
            "entities": self.entities,
            "session_id": self.session_id,
            "execution_time": self.execution_time,
            "error": self.error,
            "timestamp": self.timestamp,
        }


class ChatWorkflow:
    """
    Chat Workflow

    Orchestrates the chat query processing flow using dependency injection.

    Usage:
        workflow = ChatWorkflow(chatbot=chatbot_agent, retriever=retriever)
        result = await workflow.execute(
            query="LANEIGE 순위 알려줘",
            session_id="session-123"
        )
    """

    def __init__(
        self,
        chatbot: ChatbotAgentProtocol,
        retriever: RetrieverProtocol,
        query_analyzer: QueryAnalyzer | None = None,
    ):
        """
        Args:
            chatbot: Chatbot agent implementation
            retriever: Retriever implementation
            query_analyzer: Optional query analyzer (creates default if None)
        """
        self.chatbot = chatbot
        self.retriever = retriever
        self.query_analyzer = query_analyzer or QueryAnalyzer()

    async def execute(
        self,
        query: str,
        session_id: str | None = None,
        current_metrics: dict[str, Any] | None = None,
    ) -> ChatWorkflowResult:
        """
        Execute chat workflow.

        Args:
            query: User query
            session_id: Optional session ID for conversation continuity
            current_metrics: Optional current metrics data

        Returns:
            ChatWorkflowResult with response and metadata
        """
        start_time = time.time()

        # Step 1: Analyze query
        analysis = self.query_analyzer.analyze(query)

        result = ChatWorkflowResult(
            query=query,
            complexity=analysis["complexity"].value,
            intent=analysis["intent"].value,
            session_id=session_id,
        )

        try:
            # Step 2: Retrieve context
            # Note: Retriever might not be used for all query types,
            # but we call it to gather potential context
            await self.retriever.retrieve(query=query, current_metrics=current_metrics, top_k=5)

            # Step 3: Generate response via chatbot
            chatbot_result = await self.chatbot.chat(
                query=query,
                session_id=session_id,
                current_metrics=current_metrics,
            )

            # Step 4: Populate result
            result.response = chatbot_result.get("response")
            result.sources = chatbot_result.get("sources", [])
            result.confidence = chatbot_result.get("confidence", 0.0)
            result.suggestions = chatbot_result.get("suggestions", [])
            result.entities = chatbot_result.get("entities", {})

        except Exception as e:
            result.error = str(e)
            result.response = None

        result.execution_time = time.time() - start_time
        return result
