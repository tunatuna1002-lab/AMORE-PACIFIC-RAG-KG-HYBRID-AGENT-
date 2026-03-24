#!/bin/bash
#
# AMORE Daily Crawl — launchd 해제 스크립트
#
set -euo pipefail

PLIST_NAME="com.amore.daily-crawl.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/$PLIST_NAME"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "=========================================="
echo "  AMORE Daily Crawl — Uninstaller"
echo "=========================================="
echo ""

if [ ! -f "$PLIST_DEST" ]; then
    echo -e "${RED}❌ 설치되어 있지 않습니다.${NC}"
    exit 0
fi

# 언로드
echo "🔄 launchd 언로드..."
launchctl unload "$PLIST_DEST" 2>/dev/null || true

if launchctl list | grep -q "com.amore.daily-crawl" 2>/dev/null; then
    echo -e "  ${RED}⚠️  언로드 실패 — 수동으로 해주세요: launchctl unload $PLIST_DEST${NC}"
else
    echo -e "  ${GREEN}✅${NC} 언로드 완료"
fi

# plist 삭제
echo "🗑️  plist 삭제..."
rm -f "$PLIST_DEST"
echo -e "  ${GREEN}✅${NC} $PLIST_DEST 삭제됨"

echo ""
echo -e "${GREEN}해제 완료!${NC} 매일 크롤링이 더 이상 실행되지 않습니다."
echo "로그 파일은 유지됩니다 (수동 삭제 필요)."
echo ""
