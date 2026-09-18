#!/usr/bin/env python3
"""
Railway 배포용 시작 스크립트

환경변수 PORT를 안전하게 읽어서 uvicorn 서버를 시작합니다.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn

logger = logging.getLogger(__name__)


def _build_chroma_index() -> None:
    """서버 시작 전 Chroma 색인을 한 번 최신 상태로 맞춘다.

    DocumentRetriever.initialize()는 읽기 전용으로 바뀌어(트랙 0-A) 더 이상
    색인을 쓰지 않는다. Railway 컨테이너는 `/app/data/chroma`가 배포마다
    영속되지 않아, 지금까지는 서버가 처음 초기화될 때 색인까지 함께
    수행해 왔다 — 그 배포 동작(첫 요청/초기화 시점에 색인이 준비돼 있는 것)을
    유지하기 위해 uvicorn을 띄우기 전에 같은 프로세스에서 한 번 색인한다.
    실패해도 서버 시작은 막지 않는다 (initialize()가 컬렉션 부재를 감지해
    호출부에서 별도로 처리한다).
    """
    try:
        from src.rag.build_index import build_index

        status = asyncio.run(build_index())
        print(f"Chroma index ready: {status}")
    except Exception:
        logger.exception("Chroma 인덱스 빌드 실패 — 서버는 계속 시작합니다.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Railway에서 제공하는 PORT 환경변수 사용 (기본값 8001)
    port = int(os.environ.get("PORT", 8001))

    _build_chroma_index()

    print(f"Starting server on port {port}...")

    uvicorn.run(
        "src.api.dashboard_api:app",
        host="0.0.0.0",
        port=port,
        # Railway에서는 reload 비활성화 (프로덕션 환경)
        reload=False,
        # 워커 수 (Railway Free/Hobby는 리소스 제한으로 1개 권장)
        workers=1,
        # 로그 레벨
        log_level="info",
        # 접속 로그 활성화
        access_log=True,
    )
