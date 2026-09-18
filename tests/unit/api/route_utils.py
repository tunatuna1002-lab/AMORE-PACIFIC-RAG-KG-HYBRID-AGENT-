"""FastAPI 앱에 등록된 경로를 버전과 무관하게 모으는 테스트 헬퍼.

FastAPI 0.141+ 는 include_router 결과를 `app.routes` 에 경로 없는 내부 객체
(`_IncludedRouter`)로 남기므로 `route.path for route in app.routes` 로는 API 경로가
보이지 않는다. 공개 API 인 `app.openapi()["paths"]` 로 API 경로를 모으고,
`app.routes` 에서 `.path` 를 가진 항목(Mount·문서 경로 등)을 더한다.
"""

from fastapi import FastAPI


def collect_app_paths(app: FastAPI) -> set[str]:
    """앱에 등록된 API 경로와 Mount 경로를 모두 반환."""
    paths = set(app.openapi().get("paths", {}))
    paths.update(route.path for route in app.routes if hasattr(route, "path"))
    return paths
