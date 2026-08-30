"""
API Routes Package
==================
FastAPI 라우터 모듈 (dashboard_api.py에서 분리)

구조:
- routes/chat.py: 챗봇 API (/api/v4/chat)
- routes/data.py: 데이터 API (/api/data, /api/historical)
- routes/crawl.py: 크롤링 API (/api/crawl/*)
- routes/brain.py: Brain API (/api/v4/brain/*)
- routes/export.py: 내보내기 API (/api/export/*)
- routes/deals.py: Deals API (/api/deals/*)
- routes/alerts.py: 알림 API (/api/alerts/*, /api/v3/alert-settings/*)
- dependencies.py: 공통 의존성 (인증, 레이트리밋 등)
"""
