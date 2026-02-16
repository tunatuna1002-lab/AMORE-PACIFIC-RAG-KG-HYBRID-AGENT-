"""
Unit tests for StorageAgent
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.storage_agent import StorageAgent


@pytest.fixture
def mock_sheets():
    """Mock SheetsWriter"""
    sheets = AsyncMock()
    sheets.append_rank_records = AsyncMock(return_value={"success": True})
    sheets.upsert_products_batch = AsyncMock(return_value={"created": 2, "updated": 1})
    sheets._append_row = MagicMock()
    sheets.get_rank_history = MagicMock(return_value=[])
    return sheets


@pytest.fixture
def mock_sqlite():
    """Mock SQLiteStorage"""
    sqlite = AsyncMock()
    sqlite.initialize = AsyncMock()
    sqlite.append_rank_records = AsyncMock(return_value={"success": True, "rows_added": 3})
    sqlite.save_competitor_products = AsyncMock(return_value={"success": True, "rows_added": 2})
    return sqlite


@pytest.fixture
def agent(mock_sheets, mock_sqlite):
    """Create StorageAgent with mocked dependencies"""
    with (
        patch("src.agents.storage_agent.SheetsWriter", return_value=mock_sheets),
        patch("src.agents.storage_agent.get_sqlite_storage", return_value=mock_sqlite),
    ):
        a = StorageAgent(spreadsheet_id="test-id", enable_sqlite=True)
        a.sheets = mock_sheets
        a.sqlite = mock_sqlite
        return a


@pytest.fixture
def sample_crawl_data():
    """Sample crawl data"""
    return {
        "categories": {
            "lip_care": {
                "rank_records": [
                    {"rank": 1, "brand": "LANEIGE", "asin": "B084RGF8YJ"},
                    {"rank": 3, "brand": "COSRX", "asin": "B0ABCDEF01"},
                ]
            }
        },
        "all_products": [
            {"asin": "B084RGF8YJ", "title": "LANEIGE Lip Mask", "brand": "LANEIGE"},
        ],
    }


class TestStorageAgent:
    """Test StorageAgent functionality"""

    @pytest.mark.asyncio
    async def test_execute_success(self, agent, sample_crawl_data):
        """Test successful execution"""
        result = await agent.execute(sample_crawl_data)

        assert result["status"] == "completed"
        assert result["raw_records"] == 2
        assert result["products_upserted"] == 3  # 2 created + 1 updated
        assert result["errors"] == []

    @pytest.mark.asyncio
    async def test_execute_with_sqlite(self, agent, sample_crawl_data):
        """Test that SQLite storage is called"""
        result = await agent.execute(sample_crawl_data)

        agent.sqlite.initialize.assert_called()
        agent.sqlite.append_rank_records.assert_called_once()
        assert result["sqlite_records"] == 3

    @pytest.mark.asyncio
    async def test_execute_sheets_failure(self, agent, sample_crawl_data):
        """Test handling of Sheets write failure"""
        agent.sheets.append_rank_records = AsyncMock(
            return_value={"success": False, "error": "API error"}
        )
        result = await agent.execute(sample_crawl_data)

        assert result["raw_records"] == 0
        assert len(result["errors"]) >= 1
        assert result["errors"][0]["step"] == "raw_data_sheets"

    @pytest.mark.asyncio
    async def test_execute_sqlite_failure(self, agent, sample_crawl_data):
        """Test handling of SQLite failure (still succeeds with Sheets)"""
        agent.sqlite.append_rank_records = AsyncMock(side_effect=Exception("DB error"))
        result = await agent.execute(sample_crawl_data)

        # Sheets should still succeed
        assert result["raw_records"] == 2
        # But SQLite error should be recorded
        sqlite_errors = [e for e in result["errors"] if "sqlite" in e["step"]]
        assert len(sqlite_errors) >= 1

    @pytest.mark.asyncio
    async def test_execute_empty_data(self, agent):
        """Test execution with empty crawl data"""
        result = await agent.execute({"categories": {}})

        assert result["status"] == "completed"
        assert result["raw_records"] == 0

    @pytest.mark.asyncio
    async def test_execute_with_competitor_products(self, agent):
        """Test saving competitor product data"""
        data = {
            "categories": {},
            "all_products": [],
            "competitor_products": [
                {"asin": "B0C42HJRBF", "brand": "Summer Fridays"},
            ],
        }
        result = await agent.execute(data)

        agent.sqlite.save_competitor_products.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_partial_failure(self, agent, sample_crawl_data):
        """Test status is 'partial' when some steps fail"""
        agent.sheets.upsert_products_batch = AsyncMock(side_effect=Exception("Batch error"))
        result = await agent.execute(sample_crawl_data)

        # raw_records saved but product upsert failed
        assert result["status"] == "partial"
        assert result["raw_records"] == 2

    def test_get_results_empty(self, agent):
        """Test get_results before execution"""
        assert agent.get_results() == {}

    def test_get_historical_data(self, agent):
        """Test historical data retrieval"""
        agent.sheets.get_rank_history = MagicMock(return_value=[{"rank": 5}])
        result = agent.get_historical_data("B084RGF8YJ", days=7)
        assert result == [{"rank": 5}]
        agent.sheets.get_rank_history.assert_called_once_with("B084RGF8YJ", 7)


# =========================================================================
# Tracer / Metrics integration
# =========================================================================


class TestStorageAgentTracerMetrics:
    """tracer와 metrics가 활성화된 경우 테스트"""

    @pytest.fixture
    def agent_with_tracer(self, mock_sheets, mock_sqlite):
        """tracer + metrics 활성화된 agent"""
        tracer = MagicMock()
        metrics = MagicMock()
        with (
            patch("src.agents.storage_agent.SheetsWriter", return_value=mock_sheets),
            patch("src.agents.storage_agent.get_sqlite_storage", return_value=mock_sqlite),
        ):
            a = StorageAgent(
                spreadsheet_id="test-id",
                enable_sqlite=True,
                tracer=tracer,
                metrics=metrics,
            )
            a.sheets = mock_sheets
            a.sqlite = mock_sqlite
            return a

    @pytest.mark.asyncio
    async def test_tracer_spans_called(self, agent_with_tracer, sample_crawl_data):
        """tracer가 올바르게 호출되는지"""
        result = await agent_with_tracer.execute(sample_crawl_data)
        tracer = agent_with_tracer.tracer
        assert tracer.start_span.call_count >= 2  # storage_agent, save_raw_data, upsert_products
        assert tracer.end_span.call_count >= 2
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_metrics_recorded(self, agent_with_tracer, sample_crawl_data):
        """metrics가 기록되는지"""
        await agent_with_tracer.execute(sample_crawl_data)
        m = agent_with_tracer.metrics
        m.record_agent_start.assert_called_once_with("storage")
        m.record_agent_complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_tracer_on_competitor_data(self, agent_with_tracer):
        """competitor 데이터 저장 시 tracer 호출"""
        data = {
            "categories": {},
            "all_products": [],
            "competitor_products": [{"asin": "B0C42HJRBF", "brand": "Test"}],
        }
        await agent_with_tracer.execute(data)
        tracer = agent_with_tracer.tracer
        calls = [str(c) for c in tracer.start_span.call_args_list]
        assert any("competitor" in c for c in calls)

    @pytest.mark.asyncio
    async def test_tracer_on_raw_data_exception(self, agent_with_tracer):
        """raw data 저장 중 예외 시 tracer end_span(failed) 호출"""
        agent_with_tracer.sheets.append_rank_records = AsyncMock(
            side_effect=Exception("Sheet crash")
        )
        data = {
            "categories": {
                "lip_care": {"rank_records": [{"rank": 1, "brand": "X", "asin": "A01"}]}
            },
        }
        result = await agent_with_tracer.execute(data)
        tracer = agent_with_tracer.tracer
        # end_span should be called with "failed"
        failed_calls = [c for c in tracer.end_span.call_args_list if "failed" in str(c)]
        assert len(failed_calls) >= 1

    @pytest.mark.asyncio
    async def test_tracer_on_upsert_exception(self, agent_with_tracer, sample_crawl_data):
        """upsert 예외 시 tracer end_span(failed) 호출"""
        agent_with_tracer.sheets.upsert_products_batch = AsyncMock(
            side_effect=Exception("Upsert crash")
        )
        result = await agent_with_tracer.execute(sample_crawl_data)
        tracer = agent_with_tracer.tracer
        failed_calls = [c for c in tracer.end_span.call_args_list if "failed" in str(c)]
        assert len(failed_calls) >= 1


# =========================================================================
# Non-dict records (model_dump path)
# =========================================================================


class TestStorageAgentRecordTypes:
    """다양한 record 타입 테스트"""

    @pytest.mark.asyncio
    async def test_non_dict_record_with_model_dump(self, agent):
        """model_dump()을 가진 non-dict record 처리"""
        mock_record = MagicMock()
        mock_record.model_dump.return_value = {"rank": 1, "brand": "LANEIGE", "asin": "B001"}
        data = {"categories": {"lip_care": {"rank_records": [mock_record]}}}
        result = await agent.execute(data)
        mock_record.model_dump.assert_called_once()
        assert result["raw_records"] == 1

    @pytest.mark.asyncio
    async def test_non_dict_record_without_model_dump(self, agent):
        """model_dump() 없는 non-dict record (fallback)"""

        class SimpleRecord:
            pass

        record = SimpleRecord()
        data = {"categories": {"lip_care": {"rank_records": [record]}}}
        result = await agent.execute(data)
        # record itself is appended as-is
        assert result["raw_records"] == 1

    @pytest.mark.asyncio
    async def test_dict_and_non_dict_mixed_records(self, agent):
        """dict와 non-dict record 혼합"""
        mock_record = MagicMock()
        mock_record.model_dump.return_value = {"rank": 2, "brand": "COSRX", "asin": "B002"}
        data = {
            "categories": {
                "lip_care": {
                    "rank_records": [
                        {"rank": 1, "brand": "LANEIGE", "asin": "B001"},  # dict
                        mock_record,  # has model_dump
                    ]
                }
            }
        }
        result = await agent.execute(data)
        assert result["raw_records"] == 2


# =========================================================================
# SQLite response failure (success=False)
# =========================================================================


class TestStorageAgentSQLiteFailures:
    """SQLite 저장 실패 세부 테스트"""

    @pytest.mark.asyncio
    async def test_sqlite_append_returns_failure(self, agent, sample_crawl_data):
        """SQLite append가 success=False 반환"""
        agent.sqlite.append_rank_records = AsyncMock(
            return_value={"success": False, "error": "constraint violation"}
        )
        result = await agent.execute(sample_crawl_data)
        sqlite_errors = [e for e in result["errors"] if e["step"] == "raw_data_sqlite"]
        assert len(sqlite_errors) == 1
        assert "constraint" in sqlite_errors[0]["error"]

    @pytest.mark.asyncio
    async def test_competitor_sqlite_failure_response(self, agent):
        """competitor SQLite 저장이 success=False 반환"""
        agent.sqlite.save_competitor_products = AsyncMock(
            return_value={"success": False, "error": "table not found"}
        )
        data = {
            "categories": {},
            "all_products": [],
            "competitor_products": [{"asin": "B001", "brand": "Test"}],
        }
        result = await agent.execute(data)
        comp_errors = [e for e in result["errors"] if e["step"] == "competitor_sqlite"]
        assert len(comp_errors) == 1

    @pytest.mark.asyncio
    async def test_competitor_general_exception(self, agent):
        """competitor 저장 중 일반 예외"""
        agent.sqlite.save_competitor_products = AsyncMock(side_effect=Exception("Unexpected"))
        data = {
            "categories": {},
            "all_products": [],
            "competitor_products": [{"asin": "B001", "brand": "Test"}],
        }
        result = await agent.execute(data)
        comp_errors = [e for e in result["errors"] if e["step"] == "competitor_data"]
        assert len(comp_errors) == 1

    @pytest.mark.asyncio
    async def test_competitor_json_save_failure(self, agent):
        """competitor JSON 파일 저장 실패 (경고만, 에러 아님)"""
        agent.sqlite.save_competitor_products = AsyncMock(
            return_value={"success": True, "rows_added": 1}
        )
        data = {
            "categories": {},
            "all_products": [],
            "competitor_products": [{"asin": "B001", "brand": "Test"}],
        }
        # json.dump를 실패시킴
        with patch("builtins.open", side_effect=PermissionError("No write")):
            result = await agent.execute(data)
        # JSON 실패는 warning이므로 errors에 추가되지 않거나 competitor_data로 추가
        assert result["competitor_products_saved"] == 1

    @pytest.mark.asyncio
    async def test_sqlite_disabled(self, mock_sheets):
        """SQLite 비활성화 시 SQLite 호출 없음"""
        with patch("src.agents.storage_agent.SheetsWriter", return_value=mock_sheets):
            a = StorageAgent(spreadsheet_id="test-id", enable_sqlite=False)
            a.sheets = mock_sheets
        data = {
            "categories": {"lip_care": {"rank_records": [{"rank": 1, "brand": "X", "asin": "A01"}]}}
        }
        result = await a.execute(data)
        assert result["sqlite_records"] == 0


# =========================================================================
# Top-level exception (lines 275-285)
# =========================================================================


class TestStorageAgentTopLevelError:
    """execute 최상위 예외 테스트"""

    @pytest.mark.asyncio
    async def test_top_level_exception_raised(self, mock_sheets, mock_sqlite):
        """execute 최상위 예외 → raise (outer except, lines 275-285)"""
        tracer = MagicMock()
        metrics = MagicMock()
        with (
            patch("src.agents.storage_agent.SheetsWriter", return_value=mock_sheets),
            patch("src.agents.storage_agent.get_sqlite_storage", return_value=mock_sqlite),
        ):
            a = StorageAgent(
                spreadsheet_id="test-id",
                enable_sqlite=True,
                tracer=tracer,
                metrics=metrics,
            )
            a.sheets = mock_sheets
            a.sqlite = mock_sqlite

        # logger.agent_complete (line 267, inside try) 를 예외로 만들어 outer except 진입
        a.logger = MagicMock()
        a.logger.agent_complete = MagicMock(side_effect=RuntimeError("Fatal"))

        with pytest.raises(RuntimeError, match="Fatal"):
            await a.execute({"categories": {}})

        tracer.end_span.assert_called()
        metrics.record_agent_error.assert_called_once_with("storage", "Fatal")


# =========================================================================
# save_metrics (lines 304-372)
# =========================================================================


class TestStorageAgentSaveMetrics:
    """save_metrics 메서드 테스트"""

    @pytest.mark.asyncio
    async def test_save_brand_metrics(self, agent):
        """브랜드 지표 저장"""
        bm = MagicMock()
        bm.brand_name = "LANEIGE"
        bm.category_id = "lip_care"
        bm.share_of_shelf = 15.0
        bm.avg_rank = 3.2
        bm.product_count = 5
        bm.top10_count = 3
        bm.top20_count = 5

        result = await agent.save_metrics(brand_metrics=[bm])
        assert result["brand_metrics"] == 1
        agent.sheets._append_row.assert_called_once()
        call_args = agent.sheets._append_row.call_args
        assert call_args[0][0] == "BrandMetrics"

    @pytest.mark.asyncio
    async def test_save_product_metrics(self, agent):
        """제품 지표 저장"""
        pm = MagicMock()
        pm.asin = "B08XYZ001"
        pm.product_title = "Lip Mask"
        pm.category_id = "lip_care"
        pm.current_rank = 1
        pm.rank_change_1d = -2
        pm.rank_change_7d = 5
        pm.rank_volatility = 1.2
        pm.streak_days = 10
        pm.rating_trend = 0.01

        result = await agent.save_metrics(product_metrics=[pm])
        assert result["product_metrics"] == 1
        call_args = agent.sheets._append_row.call_args
        assert call_args[0][0] == "ProductMetrics"

    @pytest.mark.asyncio
    async def test_save_market_metrics(self, agent):
        """시장 지표 저장"""
        mm = MagicMock()
        mm.category_id = "lip_care"
        mm.hhi = 0.08
        mm.cpi = 105.2
        mm.churn_rate_7d = 12.5
        mm.avg_rating_gap = 0.3
        mm.top_brand = "LANEIGE"
        mm.top_brand_sos = 15.0

        result = await agent.save_metrics(market_metrics=[mm])
        assert result["market_metrics"] == 1
        call_args = agent.sheets._append_row.call_args
        assert call_args[0][0] == "MarketMetrics"

    @pytest.mark.asyncio
    async def test_save_all_metrics(self, agent):
        """모든 지표 동시 저장"""
        bm = MagicMock()
        bm.brand_name = "LANEIGE"
        bm.category_id = "lip_care"
        bm.share_of_shelf = 15.0
        bm.avg_rank = 3.2
        bm.product_count = 5
        bm.top10_count = 3
        bm.top20_count = 5

        pm = MagicMock()
        pm.asin = "B08XYZ001"
        pm.product_title = "Lip Mask"
        pm.category_id = "lip_care"
        pm.current_rank = 1
        pm.rank_change_1d = -2
        pm.rank_change_7d = 5
        pm.rank_volatility = 1.2
        pm.streak_days = 10
        pm.rating_trend = 0.01

        mm = MagicMock()
        mm.category_id = "lip_care"
        mm.hhi = 0.08
        mm.cpi = 105.2
        mm.churn_rate_7d = 12.5
        mm.avg_rating_gap = 0.3
        mm.top_brand = "LANEIGE"
        mm.top_brand_sos = 15.0

        result = await agent.save_metrics(
            brand_metrics=[bm], product_metrics=[pm], market_metrics=[mm]
        )
        assert result["brand_metrics"] == 1
        assert result["product_metrics"] == 1
        assert result["market_metrics"] == 1
        assert agent.sheets._append_row.call_count == 3

    @pytest.mark.asyncio
    async def test_save_metrics_none_args(self, agent):
        """None 인자 → 0 반환"""
        result = await agent.save_metrics()
        assert result["brand_metrics"] == 0
        assert result["product_metrics"] == 0
        assert result["market_metrics"] == 0
        agent.sheets._append_row.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_metrics_exception(self, agent):
        """save_metrics 예외 → raise"""
        agent.sheets._append_row = MagicMock(side_effect=Exception("API limit"))
        bm = MagicMock()
        bm.brand_name = "LANEIGE"
        bm.category_id = "lip_care"
        bm.share_of_shelf = 15.0
        bm.avg_rank = 3.2
        bm.product_count = 5
        bm.top10_count = 3
        bm.top20_count = 5

        with pytest.raises(Exception, match="API limit"):
            await agent.save_metrics(brand_metrics=[bm])

    @pytest.mark.asyncio
    async def test_save_metrics_with_tracer(self, mock_sheets, mock_sqlite):
        """save_metrics tracer 호출 확인"""
        tracer = MagicMock()
        with (
            patch("src.agents.storage_agent.SheetsWriter", return_value=mock_sheets),
            patch("src.agents.storage_agent.get_sqlite_storage", return_value=mock_sqlite),
        ):
            a = StorageAgent(spreadsheet_id="test-id", tracer=tracer)
            a.sheets = mock_sheets

        result = await a.save_metrics()
        tracer.start_span.assert_called_once_with("save_metrics")
        tracer.end_span.assert_called_once_with("completed")

    @pytest.mark.asyncio
    async def test_save_metrics_exception_with_tracer(self, mock_sheets, mock_sqlite):
        """save_metrics 예외 시 tracer end_span(failed) 호출"""
        tracer = MagicMock()
        with (
            patch("src.agents.storage_agent.SheetsWriter", return_value=mock_sheets),
            patch("src.agents.storage_agent.get_sqlite_storage", return_value=mock_sqlite),
        ):
            a = StorageAgent(spreadsheet_id="test-id", tracer=tracer)
            a.sheets = mock_sheets
            a.sheets._append_row = MagicMock(side_effect=RuntimeError("fail"))

        bm = MagicMock()
        bm.brand_name = "X"
        bm.category_id = "c"
        bm.share_of_shelf = 1.0
        bm.avg_rank = 1.0
        bm.product_count = 1
        bm.top10_count = 1
        bm.top20_count = 1

        with pytest.raises(RuntimeError):
            await a.save_metrics(brand_metrics=[bm])
        failed_calls = [c for c in tracer.end_span.call_args_list if "failed" in str(c)]
        assert len(failed_calls) >= 1

    @pytest.mark.asyncio
    async def test_save_multiple_brand_metrics(self, agent):
        """여러 브랜드 지표 저장"""
        metrics = []
        for name in ["LANEIGE", "COSRX", "innisfree"]:
            bm = MagicMock()
            bm.brand_name = name
            bm.category_id = "lip_care"
            bm.share_of_shelf = 10.0
            bm.avg_rank = 5.0
            bm.product_count = 3
            bm.top10_count = 2
            bm.top20_count = 3
            metrics.append(bm)

        result = await agent.save_metrics(brand_metrics=metrics)
        assert result["brand_metrics"] == 3
        assert agent.sheets._append_row.call_count == 3


# =========================================================================
# Status determination (failed vs partial)
# =========================================================================


class TestStorageAgentStatus:
    """상태 결정 로직 테스트"""

    @pytest.mark.asyncio
    async def test_failed_status_when_no_records_saved(self, agent):
        """raw_records=0이고 에러가 있으면 failed"""
        agent.sheets.append_rank_records = AsyncMock(
            return_value={"success": False, "error": "fail"}
        )
        data = {
            "categories": {"lip_care": {"rank_records": [{"rank": 1, "brand": "X", "asin": "A01"}]}}
        }
        result = await agent.execute(data)
        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_get_results_after_execute(self, agent, sample_crawl_data):
        """execute 후 get_results가 결과를 반환"""
        await agent.execute(sample_crawl_data)
        results = agent.get_results()
        assert results["status"] == "completed"
        assert results["raw_records"] == 2


# =========================================================================
# Additional edge cases for higher coverage
# =========================================================================


class TestStorageAgentEdgeCases:
    """추가 엣지 케이스 테스트"""

    @pytest.mark.asyncio
    async def test_execute_with_all_products_empty(self, agent, mock_sheets, mock_sqlite):
        """all_products가 빈 리스트인 경우"""
        data = {
            "categories": {
                "lip_care": {"rank_records": [{"rank": 1, "brand": "X", "asin": "A01"}]}
            },
            "all_products": [],  # Empty products
        }
        result = await agent.execute(data)
        assert result["status"] == "completed"
        assert result["products_upserted"] == 0

    @pytest.mark.asyncio
    async def test_execute_with_products_missing_fields(self, agent):
        """제품 데이터에 필수 필드가 누락된 경우"""
        data = {
            "categories": {},
            "all_products": [
                {"asin": "B001"},  # title, brand 누락
                {"title": "Product", "brand": "Brand"},  # asin 누락
            ],
        }
        result = await agent.execute(data)
        # 누락된 필드는 기본값으로 처리됨
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_execute_multiple_categories(self, agent):
        """여러 카테고리 데이터 동시 처리"""
        data = {
            "categories": {
                "lip_care": {
                    "rank_records": [
                        {"rank": 1, "brand": "LANEIGE", "asin": "B001"},
                        {"rank": 2, "brand": "COSRX", "asin": "B002"},
                    ]
                },
                "skin_care": {
                    "rank_records": [
                        {"rank": 1, "brand": "innisfree", "asin": "B003"},
                    ]
                },
                "makeup": {
                    "rank_records": [
                        {"rank": 5, "brand": "Etude House", "asin": "B004"},
                    ]
                },
            },
            "all_products": [],
        }
        result = await agent.execute(data)
        assert result["status"] == "completed"
        assert result["raw_records"] == 4  # 2 + 1 + 1

    @pytest.mark.asyncio
    async def test_sqlite_initialization_called(self, agent, sample_crawl_data):
        """SQLite initialize가 호출되는지 확인"""
        await agent.execute(sample_crawl_data)
        # initialize should be called at least once
        assert agent.sqlite.initialize.call_count >= 1

    @pytest.mark.asyncio
    async def test_competitor_products_without_sqlite(self, mock_sheets):
        """SQLite 비활성화 시 competitor products 처리"""
        with patch("src.agents.storage_agent.SheetsWriter", return_value=mock_sheets):
            a = StorageAgent(spreadsheet_id="test-id", enable_sqlite=False)
            a.sheets = mock_sheets

        data = {
            "categories": {},
            "all_products": [],
            "competitor_products": [{"asin": "B001", "brand": "Test"}],
        }
        result = await a.execute(data)
        # SQLite disabled이므로 competitor_products_saved 키가 없어야 함
        assert "competitor_products_saved" not in result

    @pytest.mark.asyncio
    async def test_product_url_generation(self, agent):
        """product_url이 없을 때 자동 생성"""
        data = {
            "categories": {},
            "all_products": [
                {"asin": "B084RGF8YJ", "title": "Lip Mask", "brand": "LANEIGE"}
                # url 필드 없음
            ],
        }
        result = await agent.execute(data)
        assert result["status"] == "completed"
        # upsert_products_batch가 호출되었고 url이 생성되었는지 확인
        agent.sheets.upsert_products_batch.assert_called_once()
        call_args = agent.sheets.upsert_products_batch.call_args[0][0]
        assert "amazon.com/dp/B084RGF8YJ" in call_args[0]["product_url"]

    @pytest.mark.asyncio
    async def test_product_with_explicit_url(self, agent):
        """명시적으로 제공된 url 사용"""
        data = {
            "categories": {},
            "all_products": [
                {
                    "asin": "B084RGF8YJ",
                    "title": "Lip Mask",
                    "brand": "LANEIGE",
                    "url": "https://www.amazon.com/custom/url",
                }
            ],
        }
        result = await agent.execute(data)
        agent.sheets.upsert_products_batch.assert_called_once()
        call_args = agent.sheets.upsert_products_batch.call_args[0][0]
        assert call_args[0]["product_url"] == "https://www.amazon.com/custom/url"

    @pytest.mark.asyncio
    async def test_logger_warnings_on_sqlite_failure(self, agent, sample_crawl_data, caplog):
        """SQLite 실패 시 동기화 불일치 경고 로그"""
        import logging

        caplog.set_level(logging.WARNING)
        agent.sqlite.append_rank_records = AsyncMock(side_effect=Exception("DB locked"))
        await agent.execute(sample_crawl_data)
        # Warning log should contain sync warning
        assert any("불일치" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_empty_category_records(self, agent):
        """카테고리는 있지만 rank_records가 빈 경우"""
        data = {
            "categories": {
                "lip_care": {"rank_records": []},
                "skin_care": {"rank_records": []},
            },
            "all_products": [],
        }
        result = await agent.execute(data)
        assert result["status"] == "completed"
        assert result["raw_records"] == 0

    @pytest.mark.asyncio
    async def test_category_without_rank_records_key(self, agent):
        """rank_records 키가 없는 카테고리"""
        data = {
            "categories": {
                "lip_care": {},  # No rank_records key
                "skin_care": {"some_other_key": "value"},
            },
            "all_products": [],
        }
        result = await agent.execute(data)
        assert result["status"] == "completed"
        assert result["raw_records"] == 0


# =========================================================================
# Initialization tests
# =========================================================================


class TestStorageAgentInit:
    """초기화 관련 테스트"""

    def test_init_with_custom_logger(self):
        """커스텀 로거로 초기화"""
        custom_logger = MagicMock()
        with (
            patch("src.agents.storage_agent.SheetsWriter"),
            patch("src.agents.storage_agent.get_sqlite_storage", return_value=None),
        ):
            agent = StorageAgent(
                spreadsheet_id="test-id", logger=custom_logger, enable_sqlite=False
            )
        assert agent.logger == custom_logger

    def test_init_without_spreadsheet_id(self):
        """spreadsheet_id 없이 초기화"""
        with (
            patch("src.agents.storage_agent.SheetsWriter") as mock_sheets_cls,
            patch("src.agents.storage_agent.get_sqlite_storage", return_value=None),
        ):
            agent = StorageAgent(enable_sqlite=False)
            mock_sheets_cls.assert_called_once_with(None)

    def test_init_with_all_monitoring_tools(self):
        """모든 모니터링 도구와 함께 초기화"""
        logger = MagicMock()
        tracer = MagicMock()
        metrics = MagicMock()
        with (
            patch("src.agents.storage_agent.SheetsWriter"),
            patch("src.agents.storage_agent.get_sqlite_storage", return_value=MagicMock()),
        ):
            agent = StorageAgent(
                spreadsheet_id="test-id",
                logger=logger,
                tracer=tracer,
                metrics=metrics,
                enable_sqlite=True,
            )
        assert agent.logger == logger
        assert agent.tracer == tracer
        assert agent.metrics == metrics


# =========================================================================
# Batch upsert edge cases
# =========================================================================


class TestStorageAgentBatchUpsert:
    """배치 업서트 엣지 케이스"""

    @pytest.mark.asyncio
    async def test_batch_upsert_with_product_name_field(self, agent):
        """product_name 필드가 있는 경우 (title 대신)"""
        data = {
            "categories": {},
            "all_products": [
                {
                    "asin": "B001",
                    "product_name": "Legacy Product Name",  # product_name instead of title
                    "brand": "Test",
                }
            ],
        }
        result = await agent.execute(data)
        agent.sheets.upsert_products_batch.assert_called_once()
        call_args = agent.sheets.upsert_products_batch.call_args[0][0]
        assert call_args[0]["product_name"] == "Legacy Product Name"

    @pytest.mark.asyncio
    async def test_batch_upsert_title_priority(self, agent):
        """title이 product_name보다 우선"""
        data = {
            "categories": {},
            "all_products": [
                {
                    "asin": "B001",
                    "title": "Title Field",
                    "product_name": "Product Name Field",
                    "brand": "Test",
                }
            ],
        }
        result = await agent.execute(data)
        call_args = agent.sheets.upsert_products_batch.call_args[0][0]
        assert call_args[0]["product_name"] == "Title Field"

    @pytest.mark.asyncio
    async def test_batch_upsert_missing_brand(self, agent):
        """brand 필드 누락 시 'Unknown' 사용"""
        data = {
            "categories": {},
            "all_products": [
                {"asin": "B001", "title": "Product"}
                # brand 누락
            ],
        }
        result = await agent.execute(data)
        call_args = agent.sheets.upsert_products_batch.call_args[0][0]
        assert call_args[0]["brand"] == "Unknown"


# =========================================================================
# Historical data retrieval
# =========================================================================


class TestStorageAgentHistoricalData:
    """히스토리 데이터 조회 테스트"""

    def test_get_historical_data_with_different_days(self, agent):
        """다양한 일수로 히스토리 조회"""
        agent.sheets.get_rank_history = MagicMock(return_value=[{"rank": i} for i in range(1, 91)])
        result = agent.get_historical_data("B084RGF8YJ", days=90)
        assert len(result) == 90
        agent.sheets.get_rank_history.assert_called_once_with("B084RGF8YJ", 90)

    def test_get_historical_data_default_days(self, agent):
        """기본 일수(30일)로 히스토리 조회"""
        agent.sheets.get_rank_history = MagicMock(return_value=[])
        result = agent.get_historical_data("B084RGF8YJ")
        agent.sheets.get_rank_history.assert_called_once_with("B084RGF8YJ", 30)
