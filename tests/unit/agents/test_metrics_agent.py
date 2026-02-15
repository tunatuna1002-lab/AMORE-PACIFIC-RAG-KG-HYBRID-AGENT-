"""
Unit tests for MetricsAgent
"""

import json
import tempfile

import pytest

from src.agents.metrics_agent import MetricsAgent


@pytest.fixture
def config_file():
    """Create a temporary thresholds config file"""
    config = {
        "ranking": {"top_n_tiers": [3, 5, 10, 20, 50, 100], "significant_drop": 5},
        "brand_health": {"sos_change_up": 1.0, "sos_change_down": -1.0},
        "thresholds": {"significant_rank_drop": 5},
        "categories": {},
        "target_brands": ["LANEIGE"],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        return f.name


@pytest.fixture
def agent(config_file):
    """Create MetricsAgent with temp config"""
    return MetricsAgent(config_path=config_file)


@pytest.fixture
def sample_crawl_data():
    """Sample crawl data with products"""
    return {
        "categories": {
            "lip_care": {
                "rank_records": [
                    {
                        "rank": 1,
                        "brand": "LANEIGE",
                        "title": "LANEIGE Lip Sleeping Mask",
                        "product_asin": "B084RGF8YJ",
                        "price": 24.0,
                        "rating": 4.7,
                        "review_count": 85000,
                    },
                    {
                        "rank": 3,
                        "brand": "Summer Fridays",
                        "title": "Summer Fridays Lip Butter Balm",
                        "product_asin": "B0C42HJRBF",
                        "price": 22.0,
                        "rating": 4.5,
                        "review_count": 12000,
                    },
                    {
                        "rank": 5,
                        "brand": "COSRX",
                        "title": "COSRX Lip Balm",
                        "product_asin": "B0ABCDEF01",
                        "price": 12.0,
                        "rating": 4.3,
                        "review_count": 5000,
                    },
                ]
            }
        }
    }


class TestMetricsAgent:
    """Test MetricsAgent functionality"""

    def test_init(self, agent):
        """Test agent initialization"""
        assert agent.config is not None
        assert agent.calculator is not None
        assert agent._results == {}

    @pytest.mark.asyncio
    async def test_execute_with_crawl_data(self, agent, sample_crawl_data):
        """Test execute with sample crawl data"""
        result = await agent.execute(sample_crawl_data)

        assert result["status"] == "completed"
        assert len(result["market_metrics"]) == 1
        assert len(result["brand_metrics"]) >= 1
        assert "summary" in result
        assert "calculated_at" in result

    @pytest.mark.asyncio
    async def test_execute_with_empty_data(self, agent):
        """Test execute with empty categories"""
        result = await agent.execute({"categories": {}})

        assert result["status"] == "completed"
        assert result["market_metrics"] == []
        assert result["brand_metrics"] == []
        assert result["product_metrics"] == []

    @pytest.mark.asyncio
    async def test_execute_with_no_rank_records(self, agent):
        """Test execute with categories but no rank_records"""
        data = {"categories": {"lip_care": {"rank_records": []}}}
        result = await agent.execute(data)

        assert result["status"] == "completed"
        assert result["market_metrics"] == []

    def test_calculate_market_metrics(self, agent):
        """Test market metrics calculation"""
        products = [
            {"rank": 1, "brand": "LANEIGE", "rating": 4.7, "price": 24.0},
            {"rank": 3, "brand": "Summer Fridays", "rating": 4.5, "price": 22.0},
            {"rank": 5, "brand": "COSRX", "rating": 4.3, "price": 12.0},
        ]
        result = agent._calculate_market_metrics("lip_care", products)

        assert result["category_id"] == "lip_care"
        assert result["total_products"] == 3
        assert result["hhi"] is not None
        assert result["top_brand"] is not None

    def test_calculate_brand_metrics(self, agent):
        """Test brand metrics calculation"""
        products = [
            {"rank": 1, "brand": "LANEIGE"},
            {"rank": 2, "brand": "LANEIGE"},
            {"rank": 5, "brand": "COSRX"},
        ]
        result = agent._calculate_brand_metrics("lip_care", products)

        assert len(result) == 2  # 2 brands
        laneige = next(b for b in result if b["brand_name"] == "LANEIGE")
        assert laneige["product_count"] == 2
        assert laneige["is_laneige"] is True

    def test_calculate_product_metrics(self, agent):
        """Test product metrics calculation"""
        product = {"rank": 3, "product_asin": "B084RGF8YJ", "title": "LANEIGE Lip Mask"}
        history = [{"rank": 5}, {"rank": 4}]

        result = agent._calculate_product_metrics(product, history, "lip_care")

        assert result["asin"] == "B084RGF8YJ"
        assert result["current_rank"] == 3
        assert result["rank_change_1d"] == -1  # improved from 4 to 3
        assert result["category_id"] == "lip_care"

    def test_check_alerts_rank_drop(self, agent):
        """Test alert generation on rank drop"""
        product_metric = {
            "asin": "B084RGF8YJ",
            "product_title": "LANEIGE Lip Mask",
            "current_rank": 15,
            "rank_change_1d": 10,
            "category_id": "lip_care",
        }
        alerts = agent._check_alerts(product_metric, {}, [])

        assert len(alerts) >= 1
        rank_drop = next(a for a in alerts if a["type"] == "rank_drop")
        assert rank_drop["severity"] == "critical"  # 10 >= 10

    def test_check_alerts_top10_entry(self, agent):
        """Test alert on Top 10 entry"""
        product_metric = {
            "asin": "B084RGF8YJ",
            "product_title": "LANEIGE Lip Mask",
            "current_rank": 8,
            "rank_change_1d": -5,  # improved from 13 to 8
            "category_id": "lip_care",
        }
        alerts = agent._check_alerts(product_metric, {}, [])

        top10 = [a for a in alerts if a["type"] == "top10_entry"]
        assert len(top10) == 1

    def test_is_laneige(self, agent):
        """Test LANEIGE brand detection"""
        assert agent._is_laneige({"brand": "LANEIGE", "title": "Lip Mask"})
        assert agent._is_laneige({"brand": "Other", "title": "LANEIGE Lip Sleeping Mask"})
        assert not agent._is_laneige({"brand": "COSRX", "title": "Lip Butter"})

    def test_calc_avg_rating_gap(self, agent):
        """Test average rating gap calculation"""
        products = [{"rating": 4.8}, {"rating": 4.0}, {"rating": 4.5}]
        gap = agent._calc_avg_rating_gap(products)
        assert gap == 0.8  # 4.8 - 4.0

    def test_calc_avg_rating_gap_empty(self, agent):
        """Test rating gap with no ratings"""
        assert agent._calc_avg_rating_gap([]) is None
        assert agent._calc_avg_rating_gap([{"rating": None}]) is None

    def test_generate_summary(self, agent):
        """Test summary generation"""
        results = {
            "brand_metrics": [
                {
                    "brand_name": "LANEIGE",
                    "is_laneige": True,
                    "category_id": "lip_care",
                    "share_of_shelf": 0.1,
                }
            ],
            "product_metrics": [
                {"asin": "B084RGF8YJ", "product_title": "Lip Mask", "current_rank": 1}
            ],
            "alerts": [{"severity": "warning"}, {"severity": "critical"}],
        }
        summary = agent._generate_summary(results)

        assert summary["laneige_products_tracked"] == 1
        assert summary["alert_count"] == 2
        assert summary["critical_alerts"] == 1
        assert summary["best_ranking_product"]["rank"] == 1

    def test_get_results_empty(self, agent):
        """Test get_results before execution"""
        assert agent.get_results() == {}
