#!/bin/bash
#
# AMORE Daily Crawl — launchd 설치 스크립트
#
# Usage:
#   ./scripts/launchd/install.sh          # 설치
#   ./scripts/launchd/install.sh --status # 상태 확인
#   ./scripts/launchd/install.sh --test   # 즉시 한번 실행 (테스트)
#
set -euo pipefail

# ── 경로 설정 ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PLIST_TEMPLATE="$SCRIPT_DIR/com.amore.daily-crawl.plist"
PLIST_NAME="com.amore.daily-crawl.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/$PLIST_NAME"
PYTHON_PATH="/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13"

# ── 색상 ──
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "=========================================="
echo "  AMORE Daily Crawl — launchd Installer"
echo "=========================================="
echo ""

# ── --status 모드 ──
if [[ "${1:-}" == "--status" ]]; then
    echo "📋 상태 확인..."
    echo ""

    if [ -f "$PLIST_DEST" ]; then
        echo -e "${GREEN}✅ plist 설치됨${NC}: $PLIST_DEST"
    else
        echo -e "${RED}❌ plist 미설치${NC}"
        exit 1
    fi

    if launchctl list | grep -q "com.amore.daily-crawl"; then
        echo -e "${GREEN}✅ launchd 로드됨${NC}"
        launchctl list | grep "com.amore.daily-crawl"
    else
        echo -e "${YELLOW}⚠️  launchd 미로드${NC} (로그아웃/재부팅 후 자동 로드)"
    fi

    echo ""
    echo "📜 최근 로그:"
    tail -5 "$PROJECT_ROOT/logs/daily_crawl_"*.log 2>/dev/null || echo "  (로그 없음)"
    echo ""
    tail -5 "$PROJECT_ROOT/logs/launchd_stderr.log" 2>/dev/null || echo "  (stderr 로그 없음)"
    exit 0
fi

# ── --test 모드 ──
if [[ "${1:-}" == "--test" ]]; then
    echo "🧪 테스트 실행 (dry-run)..."
    echo ""
    "$PYTHON_PATH" "$PROJECT_ROOT/scripts/daily_crawl.py" --dry-run
    exit $?
fi

# ── 사전 검증 ──
echo "🔍 사전 검증..."

# Python 확인
if [ ! -x "$PYTHON_PATH" ]; then
    echo -e "${RED}❌ Python 없음: $PYTHON_PATH${NC}"
    echo "   Python 3.13을 설치해주세요."
    exit 1
fi
echo -e "  ${GREEN}✅${NC} Python: $PYTHON_PATH"

# 의존성 확인
if ! "$PYTHON_PATH" -c "import playwright, fastapi, aiosqlite" 2>/dev/null; then
    echo -e "${RED}❌ 필수 패키지 미설치${NC}"
    echo "   pip install -r requirements.txt 를 실행해주세요."
    exit 1
fi
echo -e "  ${GREEN}✅${NC} 의존성: OK"

# Playwright 브라우저 확인
if ! "$PYTHON_PATH" -c "from playwright.sync_api import sync_playwright; p=sync_playwright().start(); b=p.chromium.launch(headless=True); b.close(); p.stop(); print('OK')" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Playwright 브라우저 미설치. 설치 중...${NC}"
    "$PYTHON_PATH" -m playwright install chromium
fi
echo -e "  ${GREEN}✅${NC} Playwright Chromium: OK"

# .env 확인
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo -e "${YELLOW}⚠️  .env 파일 없음 — .env.example을 복사하세요${NC}"
fi

# 로그 디렉토리
mkdir -p "$PROJECT_ROOT/logs"
echo -e "  ${GREEN}✅${NC} 로그 디렉토리: $PROJECT_ROOT/logs/"

# ── plist 생성 ──
echo ""
echo "📝 plist 생성..."

# 기존 plist가 있으면 먼저 언로드
if launchctl list | grep -q "com.amore.daily-crawl" 2>/dev/null; then
    echo "  기존 작업 언로드..."
    launchctl unload "$PLIST_DEST" 2>/dev/null || true
fi

# 템플릿에서 경로 치환
mkdir -p "$HOME/Library/LaunchAgents"
sed \
    -e "s|__PYTHON_PATH__|$PYTHON_PATH|g" \
    -e "s|__PROJECT_ROOT__|$PROJECT_ROOT|g" \
    -e "s|__HOME__|$HOME|g" \
    "$PLIST_TEMPLATE" > "$PLIST_DEST"

echo -e "  ${GREEN}✅${NC} $PLIST_DEST"

# ── launchd 로드 ──
echo ""
echo "🚀 launchd 로드..."
launchctl load "$PLIST_DEST"

if launchctl list | grep -q "com.amore.daily-crawl"; then
    echo -e "  ${GREEN}✅ 로드 성공!${NC}"
else
    echo -e "  ${YELLOW}⚠️  로드는 했지만 아직 목록에 없음 (정상 — 다음 실행 시간에 활성화)${NC}"
fi

# ── 완료 ──
echo ""
echo "=========================================="
echo -e "  ${GREEN}설치 완료!${NC}"
echo "=========================================="
echo ""
echo "  📅 실행 시간: 매일 22:00 KST"
echo "  📂 프로젝트:  $PROJECT_ROOT"
echo "  🐍 Python:    $PYTHON_PATH"
echo "  📜 로그:      $PROJECT_ROOT/logs/daily_crawl_*.log"
echo "  📋 plist:     $PLIST_DEST"
echo ""
echo "유용한 명령어:"
echo "  상태 확인:    ./scripts/launchd/install.sh --status"
echo "  테스트 실행:  ./scripts/launchd/install.sh --test"
echo "  즉시 실행:    $PYTHON_PATH $PROJECT_ROOT/scripts/daily_crawl.py"
echo "  해제:         ./scripts/launchd/uninstall.sh"
echo ""
