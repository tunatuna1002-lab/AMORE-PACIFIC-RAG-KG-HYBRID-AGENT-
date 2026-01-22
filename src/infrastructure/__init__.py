"""
Infrastructure Layer
====================
Clean Architecture의 Frameworks & Drivers Layer

외부 서비스, 데이터베이스, 프레임워크와의 통합을 담당합니다.
Domain의 Protocol들을 구현합니다.

구조:
- config/: 설정 관리
- persistence/: 데이터 저장소 구현 (Sheets, SQLite)
- external/: 외부 서비스 (Amazon Scraper, LLM Client)
- bootstrap.py: DI Container
"""

from src.infrastructure.config.config_manager import AppConfig

__all__ = ["AppConfig"]
