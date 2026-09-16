"""
API dependency modules
======================
``src.api.dependencies`` re-exports everything here for backwards compatibility;
new code should import from the specific module:

- ``auth``        API-key verification, rate limiter, JWT e-mail tokens
- ``session``     conversation memory (one ``ConversationMemory`` for the API)
- ``audit``       chat audit trail (file handler created lazily)
- ``data``        dashboard data access (thin wrapper over DashboardDataService)
- ``suggestions`` follow-up question suggestions
- ``providers``   lazily-built singletons (RAG router/retriever, state manager, ...)
"""
