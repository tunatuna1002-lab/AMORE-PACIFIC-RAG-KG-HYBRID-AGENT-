"""
Market Intelligence Routes
==========================
4-Layer Market Intelligence 시스템 엔드포인트
"""

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.dependencies import get_market_intelligence, verify_api_key
from src.tools.storage.sqlite_storage import get_sqlite_storage

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Market Intelligence"])


# ============= Pydantic Models =============


class MarketIntelligenceStatusResponse(BaseModel):
    """Market Intelligence 상태 응답"""

    initialized: bool
    layers_collected: list[int]
    last_collection: str | None = None
    stats: dict[str, Any]


class LayerDataResponse(BaseModel):
    """레이어 데이터 응답"""

    layer: int
    layer_name: str
    collected_at: str
    data: dict[str, Any]
    sources: list[dict[str, Any]]


# ============= Endpoints =============


@router.get("/api/market-intelligence/status", response_model=MarketIntelligenceStatusResponse)
async def get_market_intelligence_status():
    """
    Market Intelligence 시스템 상태 조회

    Returns:
        초기화 상태, 수집된 레이어, 통계
    """
    try:
        engine = await get_market_intelligence()
        stats = engine.get_stats()

        # 마지막 수집 시간
        last_collection = None
        if engine.layer_data:
            times = [ld.collected_at for ld in engine.layer_data.values()]
            if times:
                last_collection = max(times)

        return MarketIntelligenceStatusResponse(
            initialized=engine._initialized,
            layers_collected=list(engine.layer_data.keys()),
            last_collection=last_collection,
            stats=stats,
        )
    except Exception as e:
        logger.error(f"Market Intelligence status error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/api/market-intelligence/layers")
async def get_market_intelligence_layers(layer: int | None = None):
    """
    4-Layer 데이터 조회

    Args:
        layer: 특정 레이어만 조회 (1-4, None이면 전체)

    Returns:
        레이어별 데이터
    """
    try:
        engine = await get_market_intelligence()

        if layer is not None:
            layer_data = engine.layer_data.get(layer)
            if not layer_data:
                return {
                    "error": f"Layer {layer} 데이터가 없습니다.",
                    "available_layers": list(engine.layer_data.keys()),
                }

            return {
                "layer": layer_data.layer,
                "layer_name": layer_data.layer_name,
                "collected_at": layer_data.collected_at,
                "data": layer_data.data,
                "sources": layer_data.sources,
            }

        # 전체 레이어
        result = {}
        for layer_num, layer_data in engine.layer_data.items():
            result[f"layer_{layer_num}"] = {
                "layer": layer_data.layer,
                "layer_name": layer_data.layer_name,
                "collected_at": layer_data.collected_at,
                "data": layer_data.data,
                "sources": layer_data.sources,
            }

        return result
    except Exception as e:
        logger.error(f"Market Intelligence layers error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/api/market-intelligence/collect", dependencies=[Depends(verify_api_key)])
async def collect_market_intelligence(layers: list[int] | None = None):
    """
    Market Intelligence 데이터 수집 트리거

    Args:
        layers: 수집할 레이어 목록 (None이면 전체)

    Returns:
        수집 결과
    """
    try:
        engine = await get_market_intelligence()

        if layers:
            # 특정 레이어만 수집
            results = {}
            for layer_num in layers:
                layer_data = await engine.collect_layer(layer_num)
                if layer_data:
                    results[f"layer_{layer_num}"] = {
                        "status": "collected",
                        "collected_at": layer_data.collected_at,
                        "sources_count": len(layer_data.sources),
                    }
                else:
                    results[f"layer_{layer_num}"] = {"status": "skipped"}
        else:
            # 전체 수집
            await engine.collect_all_layers()
            results = {
                f"layer_{k}": {
                    "status": "collected",
                    "collected_at": v.collected_at,
                    "sources_count": len(v.sources),
                }
                for k, v in engine.layer_data.items()
            }

        # 데이터 저장
        engine.save_data()

        return {"status": "success", "collected": results, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Market Intelligence collection error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/api/market-intelligence/insight")
async def get_market_intelligence_insight(include_amazon: bool = False):
    """
    4-Layer 기반 인사이트 생성

    Args:
        include_amazon: Layer 1 Amazon 데이터 포함 여부

    Returns:
        생성된 인사이트 텍스트
    """
    try:
        engine = await get_market_intelligence()

        # 데이터가 없으면 먼저 수집
        if not engine.layer_data:
            await engine.collect_all_layers()

        # Amazon 데이터 가져오기 (선택)
        amazon_data = None
        if include_amazon:
            try:
                await get_sqlite_storage()
                # 최신 LANEIGE 데이터 조회
                amazon_data = {"sos": 5.2, "laneige_rank": 15}  # placeholder
            except Exception:
                logger.warning("Suppressed Exception", exc_info=True)

        insight = engine.generate_layered_insight(amazon_data=amazon_data)

        return {
            "insight": insight,
            "generated_at": datetime.now().isoformat(),
            "layers_used": list(engine.layer_data.keys()),
        }
    except Exception as e:
        logger.error(f"Market Intelligence insight error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/api/insights/sources")
async def get_insight_sources():
    """
    인사이트 출처 정보 조회

    Returns:
        출처 목록 및 통계
    """
    try:
        engine = await get_market_intelligence()

        all_sources = []
        for layer_data in engine.layer_data.values():
            all_sources.extend(layer_data.sources)

        # 출처 유형별 통계
        by_type = {}
        for source in all_sources:
            source_type = source.get("source_type", "unknown")
            by_type[source_type] = by_type.get(source_type, 0) + 1

        return {
            "total_sources": len(all_sources),
            "by_type": by_type,
            "sources": all_sources[:20],  # 최근 20개
            "source_manager_stats": engine.source_manager.get_stats(),
        }
    except Exception as e:
        logger.error(f"Insight sources error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
