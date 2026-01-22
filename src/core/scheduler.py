"""
Autonomous Scheduler
====================
자율 작업 스케줄러 모듈

설정된 스케줄에 따라 자동 작업을 트리거합니다.
last_run 상태는 파일에 저장되어 서버 재시작 후에도 유지됩니다.

## 기본 스케줄
- daily_crawl: 매일 한국시간 06:00 크롤링
- check_data_freshness: 1시간마다 데이터 신선도 체크

## 사용 예
```python
scheduler = AutonomousScheduler()
await scheduler.start(callback=my_callback)

# 대기 중인 작업 수 확인
pending = scheduler.get_pending_count()

# 스케줄 추가
scheduler.add_schedule({
    "id": "weekly_report",
    "name": "주간 리포트",
    "action": "generate_report",
    "schedule_type": "daily",
    "hour": 9,
    "minute": 0,
    "enabled": True
})

# 스케줄러 중지
scheduler.stop()
```
"""

import logging
import json
import asyncio
import os
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Callable

logger = logging.getLogger(__name__)

# 한국 시간대 (UTC+9)
KST = timezone(timedelta(hours=9))


class AutonomousScheduler:
    """
    자율 작업 스케줄러

    설정된 스케줄에 따라 자동 작업을 트리거합니다.
    last_run 상태는 파일에 저장되어 서버 재시작 후에도 유지됩니다.
    """

    STATE_FILE = "./data/scheduler_state.json"

    def __init__(self):
        self.schedules: List[Dict[str, Any]] = []
        self._last_run: Dict[str, datetime] = {}
        self.running: bool = False
        self._task: Optional[asyncio.Task] = None
        self._load_state()
        self._load_default_schedules()

    def _load_state(self):
        """파일에서 last_run 상태 복원"""
        try:
            if os.path.exists(self.STATE_FILE):
                with open(self.STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for schedule_id, timestamp in data.get("last_run", {}).items():
                        self._last_run[schedule_id] = datetime.fromisoformat(timestamp)
                    logger.info(f"Scheduler state loaded: {list(self._last_run.keys())}")
        except Exception as e:
            logger.warning(f"Failed to load scheduler state: {e}")

    def _save_state(self):
        """last_run 상태를 파일에 원자적으로 저장 (crash-safe)"""
        try:
            os.makedirs(os.path.dirname(self.STATE_FILE), exist_ok=True)
            data = {
                "last_run": {
                    schedule_id: dt.isoformat()
                    for schedule_id, dt in self._last_run.items()
                },
                "saved_at": datetime.now().isoformat()
            }
            # 원자적 쓰기
            dir_path = os.path.dirname(self.STATE_FILE) or "."
            with tempfile.NamedTemporaryFile(mode="w", dir=dir_path, delete=False, suffix=".tmp", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                temp_path = f.name
            os.replace(temp_path, self.STATE_FILE)
        except Exception as e:
            logger.error(f"Failed to save scheduler state: {e}")
            try:
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                logger.debug(f"임시 파일 정리 실패 (무시됨): {e}")

    def _load_default_schedules(self):
        """기본 스케줄 로드"""
        self.schedules = [
            {
                "id": "daily_crawl",
                "name": "일일 크롤링",
                "action": "crawl_workflow",
                "schedule_type": "daily",
                "hour": 6,
                "minute": 0,
                "enabled": True
            },
            {
                "id": "check_data_freshness",
                "name": "데이터 신선도 체크",
                "action": "check_data",
                "schedule_type": "interval",
                "interval_hours": 1,
                "enabled": True
            }
        ]

    def get_kst_now(self) -> datetime:
        """한국 시간 기준 현재 시각 반환"""
        return datetime.now(KST)

    def get_due_tasks(self, current_time: datetime) -> List[Dict[str, Any]]:
        """실행해야 할 작업 목록 반환 (한국시간 기준)"""
        due_tasks = []
        kst_now = self.get_kst_now()
        kst_today = kst_now.date()

        for schedule in self.schedules:
            if not schedule.get("enabled", True):
                continue

            schedule_id = schedule["id"]
            last_run = self._last_run.get(schedule_id)

            if schedule["schedule_type"] == "daily":
                scheduled_time = kst_now.replace(
                    hour=schedule["hour"],
                    minute=schedule["minute"],
                    second=0,
                    microsecond=0
                )
                last_run_date = last_run.date() if last_run else None
                if (last_run_date is None or last_run_date < kst_today) and kst_now >= scheduled_time:
                    logger.info(f"Task {schedule_id} is due: last_run={last_run_date}, kst_today={kst_today}")
                    due_tasks.append(schedule)

            elif schedule["schedule_type"] == "interval":
                interval = timedelta(hours=schedule.get("interval_hours", 1))
                if last_run is None or (kst_now - last_run.replace(tzinfo=KST)) >= interval:
                    due_tasks.append(schedule)

        return due_tasks

    def mark_completed(self, schedule_id: str):
        """작업 완료 마킹 및 상태 저장"""
        self._last_run[schedule_id] = self.get_kst_now()
        self._save_state()
        logger.info(f"Marked {schedule_id} as completed at {self._last_run[schedule_id]}")

    def add_schedule(self, schedule: Dict[str, Any]):
        """스케줄 추가"""
        self.schedules.append(schedule)

    def remove_schedule(self, schedule_id: str):
        """스케줄 제거"""
        self.schedules = [s for s in self.schedules if s["id"] != schedule_id]

    def get_pending_count(self) -> int:
        """대기 중인 작업 수 반환"""
        return len(self.get_due_tasks(self.get_kst_now()))

    def stop(self):
        """스케줄러 중지"""
        self.running = False
        if self._task and not self._task.done():
            self._task.cancel()
            self._task = None

    async def start(self, callback: Callable):
        """스케줄러 시작"""
        if self.running:
            return

        self.running = True

        async def _run_loop():
            while self.running:
                try:
                    due_tasks = self.get_due_tasks(datetime.now())
                    for task in due_tasks:
                        try:
                            await callback(task)
                            self.mark_completed(task["id"])
                        except Exception as e:
                            logger.error(f"Scheduler task error: {task['id']} - {e}")
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Scheduler loop error: {e}")
                    await asyncio.sleep(60)

        self._task = asyncio.create_task(_run_loop())

    def get_status(self) -> Dict[str, Any]:
        """스케줄러 상태 반환"""
        return {
            "running": self.running,
            "schedules_count": len(self.schedules),
            "pending_tasks": self.get_pending_count(),
            "last_runs": {
                sid: dt.isoformat() for sid, dt in self._last_run.items()
            },
            "kst_now": self.get_kst_now().isoformat()
        }
