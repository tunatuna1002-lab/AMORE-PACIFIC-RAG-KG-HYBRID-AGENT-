"""
Brain Routes - UnifiedBrain API endpoints
"""
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from typing import Optional

from src.api.dependencies import verify_api_key, load_dashboard_data
from src.core.brain import UnifiedBrain, get_initialized_brain, BrainMode

router = APIRouter(prefix="/api/v4/brain", tags=["brain"])


@router.get("/status")
async def get_brain_status():
    """
    Brain 상태 조회

    Returns:
        - mode: 현재 Brain 모드
        - scheduler: 스케줄러 상태
        - pending_tasks: 대기 중 태스크
        - stats: 통계
    """
    try:
        brain = await get_initialized_brain()

        return {
            "mode": brain.mode.value,
            "scheduler_running": brain.scheduler.running if brain.scheduler else False,
            "pending_tasks": brain.scheduler.get_pending_count() if brain.scheduler else 0,
            "stats": brain.get_stats(),
            "initialized": True
        }
    except Exception as e:
        return {
            "mode": "uninitialized",
            "scheduler_running": False,
            "pending_tasks": 0,
            "stats": {},
            "initialized": False,
            "error": str(e)
        }


@router.post("/scheduler/start", dependencies=[Depends(verify_api_key)])
async def start_brain_scheduler():
    """
    자율 스케줄러 시작 (API Key 필요)

    - 일일 크롤링 (09:00)
    - 주기적 알림 체크 (30분)
    - 백그라운드 분석
    """
    try:
        brain = await get_initialized_brain()

        if brain.scheduler and brain.scheduler.running:
            return {
                "started": False,
                "message": "스케줄러가 이미 실행 중입니다.",
                "status": "running"
            }

        await brain.start_scheduler()

        return {
            "started": True,
            "message": "자율 스케줄러가 시작되었습니다.",
            "status": "running"
        }
    except Exception as e:
        return {
            "started": False,
            "message": f"스케줄러 시작 실패: {str(e)}",
            "status": "error"
        }


@router.post("/scheduler/stop", dependencies=[Depends(verify_api_key)])
async def stop_brain_scheduler():
    """자율 스케줄러 중지 (API Key 필요)"""
    try:
        brain = await get_initialized_brain()

        if brain.scheduler:
            brain.scheduler.stop()

        return {
            "stopped": True,
            "message": "스케줄러가 중지되었습니다.",
            "status": "stopped"
        }
    except Exception as e:
        return {
            "stopped": False,
            "message": f"스케줄러 중지 실패: {str(e)}",
            "status": "error"
        }


@router.post("/autonomous-cycle", dependencies=[Depends(verify_api_key)])
async def run_autonomous_cycle():
    """
    자율 사이클 수동 실행 (API Key 필요)

    1. 데이터 신선도 확인
    2. 필요시 크롤링
    3. 지표 계산
    4. 알림 조건 체크
    5. 인사이트 생성
    """
    try:
        brain = await get_initialized_brain()
        result = await brain.run_autonomous_cycle()

        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@router.post("/check-alerts")
async def check_brain_alerts():
    """
    알림 조건 수동 체크

    현재 메트릭 데이터를 기반으로 알림 조건을 체크합니다.
    """
    try:
        brain = await get_initialized_brain()
        data = load_dashboard_data()

        if not data:
            return {
                "alerts": [],
                "message": "데이터가 없습니다."
            }

        alerts = await brain.check_alerts(data)

        return {
            "alerts": alerts,
            "count": len(alerts),
            "checked_at": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "alerts": [],
            "error": str(e)
        }


@router.get("/stats")
async def get_brain_stats():
    """Brain 통계 조회"""
    try:
        brain = await get_initialized_brain()
        return brain.get_stats()
    except Exception as e:
        return {"error": str(e)}


@router.post("/mode", dependencies=[Depends(verify_api_key)])
async def set_brain_mode(mode: str):
    """
    Brain 모드 변경 (API Key 필요)

    Args:
        mode: reactive, proactive, autonomous
    """
    try:
        brain = await get_initialized_brain()

        mode_map = {
            "reactive": BrainMode.REACTIVE,
            "proactive": BrainMode.PROACTIVE,
            "autonomous": BrainMode.AUTONOMOUS
        }

        if mode not in mode_map:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode. Valid modes: {list(mode_map.keys())}"
            )

        brain.mode = mode_map[mode]

        return {
            "mode": brain.mode.value,
            "message": f"Brain 모드가 {mode}(으)로 변경되었습니다."
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}
