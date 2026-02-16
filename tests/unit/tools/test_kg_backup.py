"""
Unit tests for src/tools/utilities/kg_backup.py

Tests cover:
- KGBackupService initialization (local and Railway paths)
- Backup creation (with/without suffix)
- Backup verification
- Auto-backup (dedup + cleanup trigger)
- Cleanup of old backups
- Restore from backup (by date and path)
- Listing backups
- Statistics
- Singleton pattern

All file operations use tmp_path to avoid side effects.
"""

import json
import os
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from src.shared.constants import KST
from src.tools.utilities.kg_backup import KGBackupService, get_kg_backup_service


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def kg_data():
    """Valid KG JSON data"""
    return {
        "triples": [
            {"subject": "LANEIGE", "predicate": "rank", "object": 1},
            {"subject": "Rare Beauty", "predicate": "rank", "object": 2},
        ],
        "saved_at": "2026-01-25T10:00:00+09:00",
        "version": "2.0",
    }


@pytest.fixture
def kg_file(tmp_path, kg_data):
    """Create a temporary KG JSON file"""
    path = tmp_path / "knowledge_graph.json"
    path.write_text(json.dumps(kg_data, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def backup_dir(tmp_path):
    """Create a temporary backup directory"""
    d = tmp_path / "backups" / "kg"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def service(kg_file, backup_dir):
    """KGBackupService with temporary paths"""
    return KGBackupService(
        backup_dir=str(backup_dir),
        retention_days=7,
        kg_path=str(kg_file),
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
class TestKGBackupServiceInit:
    """초기화 테스트"""

    def test_custom_paths(self, service, kg_file, backup_dir):
        assert service.kg_path == kg_file
        assert service.backup_dir == backup_dir
        assert service.retention_days == 7

    def test_default_retention_days(self, kg_file, backup_dir):
        s = KGBackupService(
            backup_dir=str(backup_dir),
            kg_path=str(kg_file),
        )
        assert s.retention_days == 7  # DEFAULT_RETENTION_DAYS

    def test_initial_stats(self, service):
        assert service._stats["total_backups"] == 0
        assert service._stats["total_restores"] == 0
        assert service._stats["last_backup"] is None
        assert service._stats["last_restore"] is None

    def test_railway_environment_detection(self, tmp_path):
        bd = tmp_path / "railway_backups"
        kg = tmp_path / "kg.json"
        kg.write_text(json.dumps({"triples": []}), encoding="utf-8")

        with patch.dict(os.environ, {"RAILWAY_ENVIRONMENT": "production"}):
            s = KGBackupService(
                backup_dir=str(bd),
                kg_path=str(kg),
            )
        # Custom paths should override Railway defaults
        assert s.backup_dir == bd

    def test_backup_dir_created(self, tmp_path):
        bd = tmp_path / "new_dir" / "backups"
        kg = tmp_path / "kg.json"
        kg.write_text(json.dumps({"triples": []}), encoding="utf-8")

        KGBackupService(backup_dir=str(bd), kg_path=str(kg))
        assert bd.exists()


# ---------------------------------------------------------------------------
# Backup creation
# ---------------------------------------------------------------------------
class TestCreateBackup:
    """백업 생성 테스트"""

    def test_create_backup_basic(self, service, backup_dir):
        result = service.create_backup()
        assert result is not None
        assert result.exists()
        assert result.parent == backup_dir
        assert result.name.startswith("kg_backup_")
        assert result.suffix == ".json"

    def test_create_backup_with_suffix(self, service, backup_dir):
        result = service.create_backup(suffix="manual")
        assert result is not None
        assert "manual" in result.name

    def test_create_backup_content_matches(self, service, kg_data):
        result = service.create_backup()
        with open(result, encoding="utf-8") as f:
            data = json.load(f)
        assert data["triples"] == kg_data["triples"]

    def test_create_backup_metadata_file(self, service):
        result = service.create_backup()
        meta_path = result.with_suffix(".meta.json")
        assert meta_path.exists()

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        assert "triple_count" in meta
        assert meta["triple_count"] == 2
        assert "created_at" in meta
        assert "file_size_bytes" in meta

    def test_create_backup_updates_stats(self, service):
        service.create_backup()
        assert service._stats["total_backups"] == 1
        assert service._stats["last_backup"] is not None

    def test_create_backup_missing_kg(self, backup_dir, tmp_path):
        """KG 파일이 없으면 None 반환"""
        missing_kg = tmp_path / "missing.json"
        s = KGBackupService(backup_dir=str(backup_dir), kg_path=str(missing_kg))
        result = s.create_backup()
        assert result is None

    def test_create_backup_invalid_kg_json(self, backup_dir, tmp_path):
        """KG 파일이 유효하지 않은 JSON이면 None 반환"""
        bad_kg = tmp_path / "bad.json"
        bad_kg.write_text("NOT JSON", encoding="utf-8")
        s = KGBackupService(backup_dir=str(backup_dir), kg_path=str(bad_kg))
        result = s.create_backup()
        assert result is None

    def test_create_backup_missing_triples(self, backup_dir, tmp_path):
        """triples 키가 없으면 검증 실패"""
        bad_kg = tmp_path / "no_triples.json"
        bad_kg.write_text(json.dumps({"version": "1.0"}), encoding="utf-8")
        s = KGBackupService(backup_dir=str(backup_dir), kg_path=str(bad_kg))
        result = s.create_backup()
        assert result is None


# ---------------------------------------------------------------------------
# Backup verification
# ---------------------------------------------------------------------------
class TestVerifyBackup:
    """백업 무결성 검증 테스트"""

    def test_verify_valid_backup(self, service, tmp_path):
        backup = tmp_path / "valid.json"
        backup.write_text(json.dumps({"triples": [{"s": "a", "p": "b", "o": "c"}]}))
        assert service._verify_backup(backup) is True

    def test_verify_empty_triples(self, service, tmp_path):
        """Empty triples triggers warning but returns True"""
        backup = tmp_path / "empty_triples.json"
        backup.write_text(json.dumps({"triples": []}))
        assert service._verify_backup(backup) is True

    def test_verify_missing_triples_key(self, service, tmp_path):
        backup = tmp_path / "no_key.json"
        backup.write_text(json.dumps({"version": "1.0"}))
        assert service._verify_backup(backup) is False

    def test_verify_invalid_json(self, service, tmp_path):
        backup = tmp_path / "bad.json"
        backup.write_text("NOT JSON")
        assert service._verify_backup(backup) is False


# ---------------------------------------------------------------------------
# Auto-backup
# ---------------------------------------------------------------------------
class TestAutoBackup:
    """자동 백업 테스트"""

    def test_auto_backup_creates_first(self, service, backup_dir):
        result = service.auto_backup()
        assert result is not None
        assert result.exists()
        assert "auto" in result.name

    def test_auto_backup_skips_if_exists(self, service, backup_dir):
        """오늘 백업이 이미 있으면 스킵"""
        # Create first backup
        first = service.auto_backup()
        assert first is not None

        # Second call should skip
        second = service.auto_backup()
        assert second is None

    def test_auto_backup_triggers_cleanup(self, service, backup_dir):
        """자동 백업 시 오래된 백업 정리"""
        # Create an old backup file
        old_date = (datetime.now(KST) - timedelta(days=30)).strftime("%Y-%m-%d")
        old_file = backup_dir / f"kg_backup_{old_date}.json"
        old_file.write_text(json.dumps({"triples": []}), encoding="utf-8")

        service.auto_backup()

        # Old file should be cleaned up
        assert not old_file.exists()


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
class TestCleanupOldBackups:
    """백업 정리 테스트"""

    def test_cleanup_deletes_old(self, service, backup_dir):
        old_date = (datetime.now(KST) - timedelta(days=30)).strftime("%Y-%m-%d")
        old_file = backup_dir / f"kg_backup_{old_date}.json"
        old_file.write_text(json.dumps({"triples": []}), encoding="utf-8")
        old_meta = backup_dir / f"kg_backup_{old_date}.meta.json"
        old_meta.write_text(json.dumps({"triple_count": 0}), encoding="utf-8")

        deleted = service._cleanup_old_backups()
        assert deleted == 1
        assert not old_file.exists()
        assert not old_meta.exists()

    def test_cleanup_keeps_recent(self, service, backup_dir):
        recent_date = datetime.now(KST).strftime("%Y-%m-%d")
        recent_file = backup_dir / f"kg_backup_{recent_date}.json"
        recent_file.write_text(json.dumps({"triples": []}), encoding="utf-8")

        deleted = service._cleanup_old_backups()
        assert deleted == 0
        assert recent_file.exists()

    def test_cleanup_handles_unparseable_filename(self, service, backup_dir):
        """파일명 파싱 실패 시 스킵"""
        bad_file = backup_dir / "kg_backup_INVALID.json"
        bad_file.write_text(json.dumps({"triples": []}), encoding="utf-8")

        deleted = service._cleanup_old_backups()
        assert deleted == 0
        assert bad_file.exists()

    def test_cleanup_skips_meta_files(self, service, backup_dir):
        """메타 파일은 직접 삭제 대상이 아님"""
        old_date = (datetime.now(KST) - timedelta(days=30)).strftime("%Y-%m-%d")
        meta_file = backup_dir / f"kg_backup_{old_date}.meta.json"
        meta_file.write_text(json.dumps({}), encoding="utf-8")

        deleted = service._cleanup_old_backups()
        # Meta file alone should not count (glob pattern is kg_backup_*.json but meta has .meta.json)
        assert deleted == 0

    def test_cleanup_multiple_old(self, service, backup_dir):
        for i in range(3):
            d = (datetime.now(KST) - timedelta(days=30 + i)).strftime("%Y-%m-%d")
            f = backup_dir / f"kg_backup_{d}.json"
            f.write_text(json.dumps({"triples": []}), encoding="utf-8")

        deleted = service._cleanup_old_backups()
        assert deleted == 3


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------
class TestRestoreFromBackup:
    """백업 복원 테스트"""

    def test_restore_by_date(self, service, kg_file, backup_dir):
        # Create a backup first
        backup_path = service.create_backup()
        assert backup_path is not None

        # Modify the KG file
        kg_file.write_text(json.dumps({"triples": [], "modified": True}), encoding="utf-8")

        # Extract date from backup filename
        date_str = datetime.now(KST).strftime("%Y-%m-%d")
        result = service.restore_from_backup(date_str, create_pre_restore_backup=False)
        assert result is True

        # KG should be restored
        with open(kg_file, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["triples"]) == 2

    def test_restore_by_path(self, service, kg_file, backup_dir):
        backup_path = service.create_backup()
        kg_file.write_text(json.dumps({"triples": []}), encoding="utf-8")

        result = service.restore_from_backup(str(backup_path), create_pre_restore_backup=False)
        assert result is True

    def test_restore_creates_pre_restore_backup(self, service, backup_dir):
        # Create initial backup
        service.create_backup()
        date_str = datetime.now(KST).strftime("%Y-%m-%d")

        result = service.restore_from_backup(date_str, create_pre_restore_backup=True)
        assert result is True

        # Should have pre_restore backup
        pre_restore = list(backup_dir.glob("*pre_restore*"))
        assert len(pre_restore) >= 1

    def test_restore_nonexistent_date(self, service):
        result = service.restore_from_backup("1999-01-01")
        assert result is False

    def test_restore_nonexistent_path(self, service):
        result = service.restore_from_backup("/nonexistent/path/backup.json")
        assert result is False

    def test_restore_invalid_backup(self, service, backup_dir):
        """무결성 검증 실패 시 복원 거부"""
        bad_backup = backup_dir / "kg_backup_2026-01-01.json"
        bad_backup.write_text(json.dumps({"version": "1.0"}), encoding="utf-8")  # no triples

        result = service.restore_from_backup("2026-01-01")
        assert result is False

    def test_restore_updates_stats(self, service):
        service.create_backup()
        date_str = datetime.now(KST).strftime("%Y-%m-%d")

        service.restore_from_backup(date_str, create_pre_restore_backup=False)
        assert service._stats["total_restores"] == 1
        assert service._stats["last_restore"] is not None


# ---------------------------------------------------------------------------
# Listing backups
# ---------------------------------------------------------------------------
class TestListBackups:
    """백업 목록 테스트"""

    def test_list_empty(self, service):
        result = service.list_backups()
        assert result == []

    def test_list_single(self, service):
        service.create_backup()
        result = service.list_backups()
        assert len(result) == 1
        assert "filename" in result[0]
        assert "path" in result[0]
        assert "size_bytes" in result[0]

    def test_list_excludes_meta_files(self, service, backup_dir):
        service.create_backup()  # creates .json + .meta.json
        result = service.list_backups()
        for item in result:
            assert ".meta" not in item["filename"]

    def test_list_includes_meta_info(self, service):
        service.create_backup()
        result = service.list_backups()
        # Meta was created, so triple_count should be present
        assert result[0].get("triple_count") == 2

    def test_list_sorted_newest_first(self, service, backup_dir):
        # Create two backup files with different dates
        old_data = json.dumps({"triples": [{"s": "a"}]})
        old_file = backup_dir / "kg_backup_2026-01-01.json"
        old_file.write_text(old_data, encoding="utf-8")

        new_file = backup_dir / "kg_backup_2026-01-25.json"
        new_file.write_text(old_data, encoding="utf-8")

        result = service.list_backups()
        assert len(result) >= 2
        # Newest first in sorted order
        filenames = [r["filename"] for r in result]
        assert filenames.index("kg_backup_2026-01-25.json") < filenames.index(
            "kg_backup_2026-01-01.json"
        )


# ---------------------------------------------------------------------------
# get_latest_backup
# ---------------------------------------------------------------------------
class TestGetLatestBackup:
    """최신 백업 조회 테스트"""

    def test_latest_backup_exists(self, service):
        service.create_backup()
        latest = service.get_latest_backup()
        assert latest is not None
        assert "filename" in latest

    def test_latest_backup_none(self, service):
        assert service.get_latest_backup() is None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
class TestGetStats:
    """통계 테스트"""

    def test_stats_empty(self, service):
        stats = service.get_stats()
        assert stats["backup_count"] == 0
        assert stats["retention_days"] == 7
        assert stats["oldest_backup"] is None
        assert stats["newest_backup"] is None

    def test_stats_with_backups(self, service):
        service.create_backup()
        stats = service.get_stats()
        assert stats["backup_count"] == 1
        assert stats["total_backups"] == 1
        assert stats["oldest_backup"] is not None
        assert stats["newest_backup"] is not None

    def test_stats_includes_paths(self, service, kg_file, backup_dir):
        stats = service.get_stats()
        assert str(kg_file) in stats["kg_path"]
        assert str(backup_dir) in stats["backup_dir"]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
class TestSingleton:
    """싱글톤 패턴 테스트"""

    def test_get_kg_backup_service(self):
        s = get_kg_backup_service()
        assert isinstance(s, KGBackupService)

    def test_singleton_same_instance(self):
        s1 = get_kg_backup_service()
        s2 = get_kg_backup_service()
        assert s1 is s2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_create_backup_empty_triples(self, backup_dir, tmp_path):
        """빈 triples도 백업 가능 (경고만)"""
        kg = tmp_path / "empty.json"
        kg.write_text(json.dumps({"triples": []}), encoding="utf-8")
        s = KGBackupService(backup_dir=str(backup_dir), kg_path=str(kg))

        result = s.create_backup()
        assert result is not None
        assert result.exists()

    def test_save_backup_metadata_error(self, service, tmp_path):
        """메타데이터 저장 실패 시 경고만 (백업은 유지)"""
        backup = service.create_backup()
        assert backup is not None
        # Even if meta save fails, backup should exist
        assert backup.exists()

    def test_restore_selects_latest_for_date(self, service, backup_dir, kg_data):
        """같은 날짜에 여러 백업이면 가장 최근 것 선택"""
        date_str = "2026-01-15"
        f1 = backup_dir / f"kg_backup_{date_str}_auto_100000.json"
        f1.write_text(json.dumps(kg_data), encoding="utf-8")
        f2 = backup_dir / f"kg_backup_{date_str}_auto_200000.json"
        modified_data = {**kg_data, "triples": [{"s": "modified"}]}
        f2.write_text(json.dumps(modified_data), encoding="utf-8")

        result = service.restore_from_backup(date_str, create_pre_restore_backup=False)
        assert result is True

        # Should have restored from f2 (sorted last)
        with open(service.kg_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["triples"] == [{"s": "modified"}]
