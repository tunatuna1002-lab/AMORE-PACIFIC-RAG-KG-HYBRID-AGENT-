#!/usr/bin/env python3
"""
SQLite Daily Backup Script
==========================
SQLite 데이터베이스 일일 백업 스크립트

사용법:
    python scripts/backup_sqlite.py              # 백업 생성
    python scripts/backup_sqlite.py --list       # 백업 목록
    python scripts/backup_sqlite.py --restore 2026-01-27  # 복원
    python scripts/backup_sqlite.py --cleanup    # 오래된 백업 정리

환경변수:
    SQLITE_BACKUP_RETENTION_DAYS: 백업 보관 일수 (기본: 7)
"""

import argparse
import logging
import os
import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 경로 설정 (Railway Volume 자동 감지)
if Path("/data").exists():
    # Railway 환경
    DB_PATH = Path("/data/amore_data.db")
    BACKUP_DIR = Path("/data/backups/sqlite")
else:
    # 로컬 환경
    DB_PATH = Path("data/amore_data.db")
    BACKUP_DIR = Path("data/backups/sqlite")

RETENTION_DAYS = int(os.getenv("SQLITE_BACKUP_RETENTION_DAYS", "7"))


def ensure_backup_dir() -> None:
    """백업 디렉토리 생성"""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"백업 디렉토리: {BACKUP_DIR}")


def backup() -> Path | None:
    """SQLite 데이터베이스 백업 생성"""
    ensure_backup_dir()

    if not DB_PATH.exists():
        logger.error(f"데이터베이스 파일이 없습니다: {DB_PATH}")
        return None

    # 백업 파일명 (날짜 + 시간)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    backup_filename = f"amore_data_{timestamp}.db"
    backup_path = BACKUP_DIR / backup_filename

    try:
        # SQLite Online Backup API 사용 (안전한 백업)
        source_conn = sqlite3.connect(str(DB_PATH))
        backup_conn = sqlite3.connect(str(backup_path))

        with backup_conn:
            source_conn.backup(backup_conn, pages=100, progress=_backup_progress)

        source_conn.close()
        backup_conn.close()

        # 파일 크기 확인
        size_mb = backup_path.stat().st_size / (1024 * 1024)
        logger.info(f"백업 완료: {backup_path} ({size_mb:.2f} MB)")

        # 오래된 백업 자동 정리
        cleanup_old_backups()

        return backup_path

    except Exception as e:
        logger.error(f"백업 실패: {e}")
        if backup_path.exists():
            backup_path.unlink()
        return None


def _backup_progress(status: int, remaining: int, total: int) -> None:
    """백업 진행률 콜백"""
    if total > 0:
        percent = (total - remaining) / total * 100
        if remaining % 100 == 0:  # 100 페이지마다 로그
            logger.debug(f"백업 진행: {percent:.1f}%")


def list_backups() -> list[dict]:
    """백업 목록 조회"""
    ensure_backup_dir()

    backups = []
    for backup_file in sorted(BACKUP_DIR.glob("amore_data_*.db"), reverse=True):
        stat = backup_file.stat()
        backups.append(
            {
                "filename": backup_file.name,
                "path": str(backup_file),
                "size_mb": stat.st_size / (1024 * 1024),
                "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        )

    if backups:
        logger.info(f"총 {len(backups)}개 백업 발견")
        for b in backups[:5]:
            logger.info(f"  - {b['filename']} ({b['size_mb']:.2f} MB)")
    else:
        logger.warning("백업이 없습니다")

    return backups


def restore(date_or_filename: str) -> bool:
    """백업에서 복원

    Args:
        date_or_filename: 날짜 (예: 2026-01-27) 또는 전체 파일명
    """
    ensure_backup_dir()

    # 백업 파일 찾기
    backup_file = None

    if date_or_filename.endswith(".db"):
        # 전체 파일명
        backup_file = BACKUP_DIR / date_or_filename
    else:
        # 날짜로 검색 (해당 날짜의 최신 백업)
        pattern = f"amore_data_{date_or_filename}*.db"
        matches = sorted(BACKUP_DIR.glob(pattern), reverse=True)
        if matches:
            backup_file = matches[0]

    if not backup_file or not backup_file.exists():
        logger.error(f"백업을 찾을 수 없습니다: {date_or_filename}")
        return False

    try:
        # 현재 DB 백업 (복원 전)
        if DB_PATH.exists():
            pre_restore_backup = DB_PATH.with_suffix(".db.pre_restore")
            shutil.copy2(DB_PATH, pre_restore_backup)
            logger.info(f"복원 전 백업 생성: {pre_restore_backup}")

        # 복원
        shutil.copy2(backup_file, DB_PATH)
        logger.info(f"복원 완료: {backup_file} → {DB_PATH}")

        # 무결성 검사
        conn = sqlite3.connect(str(DB_PATH))
        result = conn.execute("PRAGMA integrity_check").fetchone()
        conn.close()

        if result[0] == "ok":
            logger.info("데이터베이스 무결성 검사 통과")
            return True
        else:
            logger.error(f"무결성 검사 실패: {result}")
            return False

    except Exception as e:
        logger.error(f"복원 실패: {e}")
        return False


def cleanup_old_backups() -> int:
    """오래된 백업 정리 (RETENTION_DAYS 이상)"""
    ensure_backup_dir()

    cutoff_date = datetime.now() - timedelta(days=RETENTION_DAYS)
    removed_count = 0

    for backup_file in BACKUP_DIR.glob("amore_data_*.db"):
        try:
            # 파일명에서 날짜 추출
            date_str = backup_file.stem.replace("amore_data_", "").split("_")[0]
            file_date = datetime.strptime(date_str, "%Y-%m-%d")

            if file_date < cutoff_date:
                backup_file.unlink()
                removed_count += 1
                logger.info(f"오래된 백업 삭제: {backup_file.name}")

        except (ValueError, IndexError):
            # 날짜 파싱 실패 시 무시
            continue

    if removed_count > 0:
        logger.info(f"총 {removed_count}개 오래된 백업 삭제 완료")

    return removed_count


def verify_backup(backup_path: Path) -> bool:
    """백업 파일 무결성 검증"""
    if not backup_path.exists():
        return False

    try:
        conn = sqlite3.connect(str(backup_path))
        result = conn.execute("PRAGMA integrity_check").fetchone()
        conn.close()
        return result[0] == "ok"
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="SQLite 데이터베이스 백업 관리")
    parser.add_argument("--list", action="store_true", help="백업 목록 조회")
    parser.add_argument("--restore", type=str, help="백업에서 복원 (날짜 또는 파일명)")
    parser.add_argument("--cleanup", action="store_true", help="오래된 백업 정리")
    parser.add_argument("--verify", type=str, help="백업 파일 무결성 검증")

    args = parser.parse_args()

    if args.list:
        list_backups()
    elif args.restore:
        success = restore(args.restore)
        exit(0 if success else 1)
    elif args.cleanup:
        cleanup_old_backups()
    elif args.verify:
        backup_path = Path(args.verify)
        if not backup_path.is_absolute():
            backup_path = BACKUP_DIR / args.verify
        is_valid = verify_backup(backup_path)
        logger.info(f"무결성 검증: {'통과' if is_valid else '실패'}")
        exit(0 if is_valid else 1)
    else:
        # 기본: 백업 생성
        backup()


if __name__ == "__main__":
    main()
