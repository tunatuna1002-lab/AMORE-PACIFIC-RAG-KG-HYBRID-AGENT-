"""General utility tools.

패키지 최상위 이름은 지연 로딩된다: ``src.tools.utilities`` 를 import 하는 것만으로
SQLite/Sheets 클라이언트나 KG 백업 경로가 열리지 않게 한다.
"""

import importlib

_LAZY: dict[str, str] = {
    "BrandResolver": ".brand_resolver",
    "DataIntegrityChecker": ".data_integrity_checker",
    "KGBackupService": ".kg_backup",
    "ReferenceTracker": ".reference_tracker",
}
_OPTIONAL = frozenset(())

__all__ = list(_LAZY)


def __getattr__(name: str):
    """지연 로딩: 하위 모듈은 실제로 접근할 때만 import한다."""
    if name in _LAZY:
        try:
            module = importlib.import_module(_LAZY[name], __name__)
        except ImportError:
            if name in _OPTIONAL:
                globals()[name] = None
                return None
            raise
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
