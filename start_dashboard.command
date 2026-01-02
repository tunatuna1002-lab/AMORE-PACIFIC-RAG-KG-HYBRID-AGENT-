#!/bin/bash

# AMORE Dashboard 원클릭 실행 스크립트
# 더블클릭으로 서버 시작 + 브라우저 자동 열기

cd "$(dirname "$0")"

echo "=========================================="
echo "  AMORE RAG-KG Hybrid Agent Dashboard"
echo "=========================================="
echo ""

# 기존 서버 프로세스 종료 (포트 8001 사용 중이면)
lsof -ti:8001 | xargs kill -9 2>/dev/null

echo "🚀 API 서버 시작 중..."

# 서버를 백그라운드에서 시작
python3 dashboard_api.py &
SERVER_PID=$!

# 서버가 시작될 때까지 대기
echo "⏳ 서버 준비 중..."
sleep 3

# 서버 상태 확인
if curl -s http://localhost:8001/ > /dev/null 2>&1; then
    echo "✅ 서버가 성공적으로 시작되었습니다!"
    echo ""
    echo "📊 대시보드 열기..."

    # 브라우저에서 대시보드 열기
    open "http://localhost:8001"

    echo ""
    echo "=========================================="
    echo "  서버 실행 중 (종료: Ctrl+C)"
    echo "  URL: http://localhost:8001"
    echo "=========================================="

    # 서버 프로세스 대기
    wait $SERVER_PID
else
    echo "❌ 서버 시작 실패"
    echo "로그를 확인해주세요."
fi
