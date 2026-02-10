"""
Knowledge Graph Backup Service
==============================
KG 자동 백업 및 복원 서비스

## 정책 (E2E Audit Action Item - 2026-01-27)
- 일 1회 자동 백업 (크롤링 완료 후)
- 7일 롤링 보관 (초과 시 오래된 백업 삭제)
- 백업 파일: data/backups/kg/kg_backup_{YYYY-MM-DD}.json

## 사용 예
```python
from src.tools.utilities.kg_backup import KGBackupService

backup_service = KGBackupService()

# 수동 백업
backup_path = backup_service.create_backup()

# 자동 백업 (크롤링 완료 후 호출)
backup_service.auto_backup()

# 복원
backup_service.restore_from_backup("2026-01-27")

# 백업 목록 조회
backups = backup_service.list_backups()
```
"""

import json
import logging
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 한국 시간대 (UTC+9)
KST = timezone(timedelta(hours=9))


class KGBackupService:
    """
    Knowledge Graph 백업 서비스

    - 일 1회 자동 백업
    - 7일 롤링 보관
    - 백업 무결성 검증
    """

    # 기본 설정
    DEFAULT_BACKUP_DIR = "data/backups/kg"
    DEFAULT_RETENTION_DAYS = 7
    DEFAULT_KG_PATH = "data/knowledge_graph.json"

    # Railway Volume 경로
    RAILWAY_BACKUP_DIR = "/data/backups/kg"
    RAILWAY_KG_PATH = "/data/knowledge_graph.json"

    def __init__(
        self,
        backup_dir: str | None = None,
        retention_days: int = None,
        kg_path: str | None = None,
    ):
        """
        Args:
            backup_dir: 백업 저장 디렉토리 (None이면 환경에 따라 자동 선택)
            retention_days: 백업 보관 일수 (기본 7일)
            kg_path: KG 원본 파일 경로 (None이면 환경에 따라 자동 선택)
        """
        # 프로젝트 루트 경로
        self._project_root = Path(__file__).parent.parent.parent

        # Railway 환경 감지
        self._is_railway = bool(os.environ.get("RAILWAY_ENVIRONMENT"))

        # 백업 디렉토리 (Railway Volume 우선)
        if backup_dir:
            self.backup_dir = Path(backup_dir)
        elif self._is_railway:
            self.backup_dir = Path(self.RAILWAY_BACKUP_DIR)
            logger.info("Railway environment detected, using Volume backup path: /data/backups/kg")
        else:
            self.backup_dir = self._project_root / self.DEFAULT_BACKUP_DIR
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # 보관 일수
        self.retention_days = retention_days or int(
            os.getenv("KG_BACKUP_RETENTION_DAYS", str(self.DEFAULT_RETENTION_DAYS))
        )

        # KG 원본 경로 (Railway Volume 우선)
        if kg_path:
            self.kg_path = Path(kg_path)
        elif self._is_railway:
            self.kg_path = Path(self.RAILWAY_KG_PATH)
            logger.info(
                "Railway environment detected, using Volume KG path: /data/knowledge_graph.json"
            )
        else:
            self.kg_path = self._project_root / self.DEFAULT_KG_PATH

        # 통계
        self._stats = {
            "total_backups": 0,
            "total_restores": 0,
            "last_backup": None,
            "last_restore": None,
        }

    def create_backup(self, suffix: str | None = None) -> Path | None:
        """
        KG 백업 생성

        Args:
            suffix: 백업 파일 접미사 (예: "manual", "pre_crawl")

        Returns:
            백업 파일 경로 (실패 시 None)
        """
        if not self.kg_path.exists():
            logger.warning(f"KG file not found: {self.kg_path}")
            return None

        try:
            # 백업 파일명 생성
            date_str = datetime.now(KST).strftime("%Y-%m-%d")
            time_str = datetime.now(KST).strftime("%H%M%S")

            if suffix:
                filename = f"kg_backup_{date_str}_{suffix}_{time_str}.json"
            else:
                filename = f"kg_backup_{date_str}.json"

            backup_path = self.backup_dir / filename

            # 원본 복사
            shutil.copy2(self.kg_path, backup_path)

            # 무결성 검증
            if not self._verify_backup(backup_path):
                logger.error(f"Backup verification failed: {backup_path}")
                backup_path.unlink()
                return None

            # 메타데이터 저장
            meta_path = backup_path.with_suffix(".meta.json")
            self._save_backup_metadata(backup_path, meta_path)

            # 통계 업데이트
            self._stats["total_backups"] += 1
            self._stats["last_backup"] = datetime.now(KST).isoformat()

            logger.info(f"KG backup created: {backup_path}")
            return backup_path

        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return None

    def auto_backup(self) -> Path | None:
        """
        자동 백업 (크롤링 완료 후 호출)

        - 오늘 이미 백업이 있으면 스킵
        - 보관 기간 초과 백업 정리

        Returns:
            백업 파일 경로 (스킵 시 None)
        """
        # 오늘 날짜
        today = datetime.now(KST).strftime("%Y-%m-%d")

        # 오늘 백업 확인
        existing = list(self.backup_dir.glob(f"kg_backup_{today}*.json"))
        # 메타 파일 제외
        existing = [f for f in existing if ".meta" not in str(f)]

        if existing:
            logger.info(f"Today's backup already exists: {existing[0]}")
            # 보관 기간 초과 백업만 정리
            self._cleanup_old_backups()
            return None

        # 백업 생성
        backup_path = self.create_backup(suffix="auto")

        # 오래된 백업 정리
        if backup_path:
            self._cleanup_old_backups()

        return backup_path

    def _cleanup_old_backups(self) -> int:
        """
        보관 기간 초과 백업 삭제

        Returns:
            삭제된 백업 수
        """
        cutoff = datetime.now(KST) - timedelta(days=self.retention_days)
        deleted = 0

        for backup_file in self.backup_dir.glob("kg_backup_*.json"):
            if ".meta" in str(backup_file):
                continue

            try:
                # 파일명에서 날짜 추출
                date_str = backup_file.stem.split("_")[2]  # kg_backup_YYYY-MM-DD
                file_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=KST)

                if file_date < cutoff:
                    # 백업 파일 삭제
                    backup_file.unlink()

                    # 메타 파일도 삭제
                    meta_file = backup_file.with_suffix(".meta.json")
                    if meta_file.exists():
                        meta_file.unlink()

                    deleted += 1
                    logger.info(f"Old backup deleted: {backup_file.name}")

            except (ValueError, IndexError):
                # 파일명 파싱 실패 - 스킵
                continue

        if deleted > 0:
            logger.info(f"Cleanup completed: {deleted} old backups deleted")

        return deleted

    def restore_from_backup(
        self, date_or_path: str, create_pre_restore_backup: bool = True
    ) -> bool:
        """
        백업에서 KG 복원

        Args:
            date_or_path: 날짜 (YYYY-MM-DD) 또는 전체 경로
            create_pre_restore_backup: 복원 전 현재 상태 백업 여부

        Returns:
            복원 성공 여부
        """
        # 백업 파일 찾기
        if "/" in date_or_path or "\\" in date_or_path:
            backup_path = Path(date_or_path)
        else:
            # 날짜로 검색
            matches = list(self.backup_dir.glob(f"kg_backup_{date_or_path}*.json"))
            matches = [f for f in matches if ".meta" not in str(f)]

            if not matches:
                logger.error(f"No backup found for date: {date_or_path}")
                return False

            # 가장 최근 것 선택
            backup_path = sorted(matches)[-1]

        if not backup_path.exists():
            logger.error(f"Backup file not found: {backup_path}")
            return False

        # 무결성 검증
        if not self._verify_backup(backup_path):
            logger.error(f"Backup verification failed: {backup_path}")
            return False

        try:
            # 복원 전 현재 상태 백업
            if create_pre_restore_backup and self.kg_path.exists():
                self.create_backup(suffix="pre_restore")

            # 복원
            shutil.copy2(backup_path, self.kg_path)

            # 통계 업데이트
            self._stats["total_restores"] += 1
            self._stats["last_restore"] = datetime.now(KST).isoformat()

            logger.info(f"KG restored from: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False

    def _verify_backup(self, backup_path: Path) -> bool:
        """백업 파일 무결성 검증"""
        try:
            with open(backup_path, encoding="utf-8") as f:
                data = json.load(f)

            # 필수 필드 확인
            if "triples" not in data:
                logger.error("Backup missing 'triples' field")
                return False

            # 트리플 수 확인
            triple_count = len(data.get("triples", []))
            if triple_count == 0:
                logger.warning("Backup contains 0 triples")

            return True

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in backup: {e}")
            return False
        except Exception as e:
            logger.error(f"Backup verification error: {e}")
            return False

    def _save_backup_metadata(self, backup_path: Path, meta_path: Path) -> None:
        """백업 메타데이터 저장"""
        try:
            # 원본 파일 정보
            with open(backup_path, encoding="utf-8") as f:
                data = json.load(f)

            meta = {
                "backup_file": backup_path.name,
                "created_at": datetime.now(KST).isoformat(),
                "triple_count": len(data.get("triples", [])),
                "original_saved_at": data.get("saved_at"),
                "version": data.get("version", "1.0"),
                "file_size_bytes": backup_path.stat().st_size,
            }

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save backup metadata: {e}")

    def list_backups(self) -> list[dict[str, Any]]:
        """
        백업 목록 조회

        Returns:
            백업 정보 리스트 (최신순 정렬)
        """
        backups = []

        for backup_file in sorted(self.backup_dir.glob("kg_backup_*.json"), reverse=True):
            if ".meta" in str(backup_file):
                continue

            info = {
                "filename": backup_file.name,
                "path": str(backup_file),
                "size_bytes": backup_file.stat().st_size,
                "modified_at": datetime.fromtimestamp(
                    backup_file.stat().st_mtime, tz=KST
                ).isoformat(),
            }

            # 메타데이터 로드
            meta_path = backup_file.with_suffix(".meta.json")
            if meta_path.exists():
                try:
                    with open(meta_path, encoding="utf-8") as f:
                        meta = json.load(f)
                    info["triple_count"] = meta.get("triple_count")
                    info["created_at"] = meta.get("created_at")
                except Exception:
                    pass

            backups.append(info)

        return backups

    def get_latest_backup(self) -> dict[str, Any] | None:
        """최신 백업 정보 반환"""
        backups = self.list_backups()
        return backups[0] if backups else None

    def get_stats(self) -> dict[str, Any]:
        """백업 서비스 통계"""
        backups = self.list_backups()

        return {
            **self._stats,
            "backup_count": len(backups),
            "retention_days": self.retention_days,
            "backup_dir": str(self.backup_dir),
            "kg_path": str(self.kg_path),
            "oldest_backup": backups[-1]["filename"] if backups else None,
            "newest_backup": backups[0]["filename"] if backups else None,
        }


# =============================================================================
# 싱글톤 인스턴스
# =============================================================================

_backup_service_instance: KGBackupService | None = None


def get_kg_backup_service() -> KGBackupService:
    """KGBackupService 싱글톤 반환"""
    global _backup_service_instance
    if _backup_service_instance is None:
        _backup_service_instance = KGBackupService()
    return _backup_service_instance


# =============================================================================
# CLI 인터페이스
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KG Backup Service CLI")
    parser.add_argument("command", choices=["backup", "restore", "list", "cleanup", "stats"])
    parser.add_argument("--date", help="Date for restore (YYYY-MM-DD)")
    parser.add_argument("--suffix", help="Backup suffix (e.g., manual)")
    args = parser.parse_args()

    service = KGBackupService()

    if args.command == "backup":
        path = service.create_backup(suffix=args.suffix)
        if path:
            print(f"Backup created: {path}")
        else:
            print("Backup failed")

    elif args.command == "restore":
        if not args.date:
            print("Error: --date required for restore")
        else:
            success = service.restore_from_backup(args.date)
            print("Restore successful" if success else "Restore failed")

    elif args.command == "list":
        backups = service.list_backups()
        print(f"Found {len(backups)} backups:")
        for b in backups:
            print(f"  - {b['filename']} ({b.get('triple_count', '?')} triples)")

    elif args.command == "cleanup":
        deleted = service._cleanup_old_backups()
        print(f"Deleted {deleted} old backups")

    elif args.command == "stats":
        stats = service.get_stats()
        print(json.dumps(stats, indent=2, ensure_ascii=False))
