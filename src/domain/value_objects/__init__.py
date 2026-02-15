"""
Domain Value Objects
====================
불변의 값 객체 정의

Value Object는 고유한 식별자가 없이 값으로만 정의되는 객체입니다.
동일한 속성을 가지면 동일한 것으로 간주됩니다.
"""

from .retrieval_result import UnifiedRetrievalResult

__all__ = [
    "UnifiedRetrievalResult",
]
