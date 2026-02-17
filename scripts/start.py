#!/usr/bin/env python3
"""
Railway 배포용 시작 스크립트

환경변수 PORT를 안전하게 읽어서 uvicorn 서버를 시작합니다.
"""

import os

import uvicorn

if __name__ == "__main__":
    # Railway에서 제공하는 PORT 환경변수 사용 (기본값 8001)
    port = int(os.environ.get("PORT", 8001))

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
