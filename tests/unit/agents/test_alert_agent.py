"""
Unit tests for AlertAgent (src/agents/alert_agent.py)
Coverage target: 60%+
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.alert_agent import Alert, AlertAgent, AlertPriority
from src.tools.notifications.email_sender import SendResult

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_state_manager():
    """Mock StateManager"""
    mock = MagicMock()
    mock.get_alert_recipients.return_value = ["user@test.com"]
    return mock


@pytest.fixture
def mock_rules_engine():
    """Mock RulesEngine"""
    return MagicMock()


@pytest.fixture
def mock_email_sender():
    """Mock EmailSender with AsyncMock methods"""
    mock = MagicMock()
    mock.is_enabled.return_value = True
    mock.send_rank_change_alert = AsyncMock(
        return_value=SendResult(success=True, sent_to=["user@test.com"], failed=[], message="OK")
    )
    mock.send_alert = AsyncMock(
        return_value=SendResult(success=True, sent_to=["user@test.com"], failed=[], message="OK")
    )
    mock.send_error_alert = AsyncMock(
        return_value=SendResult(success=True, sent_to=["user@test.com"], failed=[], message="OK")
    )
    mock.send_daily_summary = AsyncMock(
        return_value=SendResult(success=True, sent_to=["user@test.com"], failed=[], message="OK")
    )
    return mock


@pytest.fixture
def alert_agent(mock_state_manager, mock_rules_engine, mock_email_sender):
    """AlertAgent instance with mocked dependencies"""
    return AlertAgent(
        state_manager=mock_state_manager,
        rules_engine=mock_rules_engine,
        email_sender=mock_email_sender,
    )


# =============================================================================
# Test AlertPriority Enum
# =============================================================================


def test_alert_priority_values():
    """Test AlertPriority enum values"""
    assert AlertPriority.CRITICAL.value == 1
    assert AlertPriority.HIGH.value == 2
    assert AlertPriority.NORMAL.value == 3
    assert AlertPriority.LOW.value == 4


# =============================================================================
# Test Alert dataclass
# =============================================================================


def test_alert_creation():
    """Test Alert dataclass creation"""
    alert = Alert(
        id="test_001",
        type="rank_change",
        priority=AlertPriority.HIGH,
        title="Test Alert",
        message="Test message",
        data={"key": "value"},
    )

    assert alert.id == "test_001"
    assert alert.type == "rank_change"
    assert alert.priority == AlertPriority.HIGH
    assert alert.title == "Test Alert"
    assert alert.message == "Test message"
    assert alert.data == {"key": "value"}
    assert alert.sent is False
    assert alert.sent_to == []
    assert isinstance(alert.created_at, datetime)


def test_alert_to_dict():
    """Test Alert.to_dict() method"""
    now = datetime.now()
    alert = Alert(
        id="test_002",
        type="error",
        priority=AlertPriority.CRITICAL,
        title="Error Alert",
        message="Error occurred",
        data={"error": "test"},
        created_at=now,
        sent=True,
        sent_to=["user@test.com"],
    )

    result = alert.to_dict()

    assert result["id"] == "test_002"
    assert result["type"] == "error"
    assert result["priority"] == 1  # CRITICAL.value
    assert result["title"] == "Error Alert"
    assert result["message"] == "Error occurred"
    assert result["data"] == {"error": "test"}
    assert result["created_at"] == now.isoformat()
    assert result["sent"] is True
    assert result["sent_to"] == ["user@test.com"]


def test_alert_defaults():
    """Test Alert default values"""
    alert = Alert(
        id="test_003",
        type="test",
        priority=AlertPriority.NORMAL,
        title="Test",
        message="Test",
    )

    assert alert.data == {}
    assert alert.sent is False
    assert alert.sent_to == []
    assert isinstance(alert.created_at, datetime)


# =============================================================================
# Test AlertAgent.__init__
# =============================================================================


def test_alert_agent_init_with_all_params(mock_state_manager, mock_rules_engine, mock_email_sender):
    """Test AlertAgent initialization with all parameters"""
    agent = AlertAgent(
        state_manager=mock_state_manager,
        rules_engine=mock_rules_engine,
        email_sender=mock_email_sender,
    )

    assert agent.state_manager == mock_state_manager
    assert agent.rules_engine == mock_rules_engine
    assert agent.email_sender == mock_email_sender
    assert agent._alerts == []
    assert agent._pending_alerts == []
    assert agent._sent_alert_ids == set()
    assert agent._stats == {"total_alerts": 0, "emails_sent": 0, "emails_failed": 0}


@patch("src.core.rules_engine.RulesEngine")
@patch("src.agents.alert_agent.EmailSender")
def test_alert_agent_init_with_defaults(mock_email_cls, mock_rules_cls, mock_state_manager):
    """Test AlertAgent initialization with default RulesEngine and EmailSender"""
    mock_rules_instance = MagicMock()
    mock_email_instance = MagicMock()
    mock_rules_cls.return_value = mock_rules_instance
    mock_email_cls.return_value = mock_email_instance

    agent = AlertAgent(state_manager=mock_state_manager)

    mock_rules_cls.assert_called_once()
    mock_email_cls.assert_called_once()
    assert agent.rules_engine == mock_rules_instance
    assert agent.email_sender == mock_email_instance


# =============================================================================
# Test create_alert
# =============================================================================


def test_create_alert_basic(alert_agent):
    """Test basic alert creation"""
    alert = alert_agent.create_alert(
        alert_type="test",
        title="Test Alert",
        message="Test message",
    )

    assert alert.type == "test"
    assert alert.title == "Test Alert"
    assert alert.message == "Test message"
    assert alert.priority == AlertPriority.NORMAL
    assert alert.data == {}
    assert alert in alert_agent._alerts
    assert alert in alert_agent._pending_alerts
    assert alert_agent._stats["total_alerts"] == 1


def test_create_alert_with_all_params(alert_agent):
    """Test alert creation with all parameters"""
    data = {"key": "value", "count": 42}
    alert = alert_agent.create_alert(
        alert_type="rank_change",
        title="Rank Change",
        message="Rank dropped",
        data=data,
        priority=AlertPriority.HIGH,
    )

    assert alert.type == "rank_change"
    assert alert.title == "Rank Change"
    assert alert.message == "Rank dropped"
    assert alert.priority == AlertPriority.HIGH
    assert alert.data == data
    assert "rank_change_" in alert.id
    assert alert_agent._stats["total_alerts"] == 1


def test_create_alert_increments_stats(alert_agent):
    """Test that create_alert increments stats"""
    assert alert_agent._stats["total_alerts"] == 0

    alert_agent.create_alert("test1", "Title 1", "Message 1")
    assert alert_agent._stats["total_alerts"] == 1

    alert_agent.create_alert("test2", "Title 2", "Message 2")
    assert alert_agent._stats["total_alerts"] == 2

    alert_agent.create_alert("test3", "Title 3", "Message 3")
    assert alert_agent._stats["total_alerts"] == 3


def test_create_alert_unique_ids(alert_agent):
    """Test that each alert gets a unique ID"""
    alert1 = alert_agent.create_alert("test", "Title 1", "Message 1")
    alert2 = alert_agent.create_alert("test", "Title 2", "Message 2")
    alert3 = alert_agent.create_alert("test", "Title 3", "Message 3")

    assert alert1.id != alert2.id
    assert alert2.id != alert3.id
    assert alert1.id != alert3.id


# =============================================================================
# Test process_metrics
# =============================================================================


@pytest.mark.asyncio
async def test_process_metrics_rank_drop(alert_agent):
    """Test process_metrics detects rank drops (rank_change >= 10)"""
    metrics_data = {
        "products": [
            {
                "name": "LANEIGE Lip Sleeping Mask",
                "brand": "LANEIGE",
                "previous_rank": 5,
                "current_rank": 20,
                "rank_change": 15,
            }
        ]
    }

    alerts = await alert_agent.process_metrics(metrics_data)

    assert len(alerts) == 1
    assert alerts[0].type == "rank_change"
    assert alerts[0].priority == AlertPriority.HIGH
    assert "급락" in alerts[0].title
    assert "15등 하락" in alerts[0].message
    assert alerts[0].data["product_name"] == "LANEIGE Lip Sleeping Mask"
    assert alerts[0].data["brand"] == "LANEIGE"
    assert alerts[0].data["previous_rank"] == 5
    assert alerts[0].data["current_rank"] == 20
    assert alerts[0].data["change"] == 15


@pytest.mark.asyncio
async def test_process_metrics_rank_rise(alert_agent):
    """Test process_metrics detects rank rises (rank_change <= -10)"""
    metrics_data = {
        "products": [
            {
                "name": "Product A",
                "brand": "Brand A",
                "previous_rank": 30,
                "current_rank": 15,
                "rank_change": -15,
            }
        ]
    }

    alerts = await alert_agent.process_metrics(metrics_data)

    assert len(alerts) == 1
    assert alerts[0].type == "rank_change"
    assert alerts[0].priority == AlertPriority.NORMAL
    assert "급등" in alerts[0].title
    assert "15등 상승" in alerts[0].message
    assert alerts[0].data["change"] == -15


@pytest.mark.asyncio
async def test_process_metrics_top10_entry(alert_agent):
    """Test process_metrics detects new Top 10 entries"""
    metrics_data = {
        "products": [
            {
                "name": "Product B",
                "brand": "Brand B",
                "previous_rank": 15,
                "current_rank": 8,
                "rank_change": -7,
            }
        ]
    }

    alerts = await alert_agent.process_metrics(metrics_data)

    assert len(alerts) == 1
    assert alerts[0].type == "important_insight"
    assert alerts[0].priority == AlertPriority.HIGH
    assert "Top 10 진입" in alerts[0].title
    assert alerts[0].data["current_rank"] == 8


@pytest.mark.asyncio
async def test_process_metrics_multiple_alerts(alert_agent):
    """Test process_metrics creates multiple alerts"""
    metrics_data = {
        "products": [
            {
                "name": "Product A",
                "brand": "Brand A",
                "previous_rank": 5,
                "current_rank": 20,
                "rank_change": 15,  # Drop alert
            },
            {
                "name": "Product B",
                "brand": "Brand B",
                "previous_rank": 25,
                "current_rank": 10,
                "rank_change": -15,  # Rise alert + Top 10 alert
            },
        ]
    }

    alerts = await alert_agent.process_metrics(metrics_data)

    # Product A: 1 drop alert
    # Product B: 1 rise alert + 1 top10 alert = 2 alerts
    assert len(alerts) == 3


@pytest.mark.asyncio
async def test_process_metrics_no_alerts(alert_agent):
    """Test process_metrics with no significant changes"""
    metrics_data = {
        "products": [
            {
                "name": "Product A",
                "brand": "Brand A",
                "previous_rank": 20,
                "current_rank": 22,
                "rank_change": 2,  # Too small
            }
        ]
    }

    alerts = await alert_agent.process_metrics(metrics_data)

    assert len(alerts) == 0


@pytest.mark.asyncio
async def test_process_metrics_empty_products(alert_agent):
    """Test process_metrics with empty products list"""
    metrics_data = {"products": []}

    alerts = await alert_agent.process_metrics(metrics_data)

    assert len(alerts) == 0


@pytest.mark.asyncio
async def test_process_metrics_missing_fields(alert_agent):
    """Test process_metrics handles missing fields gracefully"""
    metrics_data = {
        "products": [
            {
                # Missing name, brand
                "previous_rank": 5,
                "current_rank": 20,
                "rank_change": 15,
            }
        ]
    }

    alerts = await alert_agent.process_metrics(metrics_data)

    assert len(alerts) == 1
    assert alerts[0].data["product_name"] is None
    assert alerts[0].data["brand"] is None


# =============================================================================
# Test send_pending_alerts
# =============================================================================


@pytest.mark.asyncio
async def test_send_pending_alerts_success(alert_agent, mock_state_manager, mock_email_sender):
    """Test successful alert sending"""
    alert_agent.create_alert("test", "Test", "Message")

    result = await alert_agent.send_pending_alerts()

    assert result["processed"] == 1
    assert result["sent"] == 1
    assert result["failed"] == 0
    assert result["skipped"] == 0
    assert len(result["details"]) == 1
    assert result["details"][0]["success"] is True
    assert len(alert_agent._pending_alerts) == 0
    assert alert_agent._stats["emails_sent"] == 1


@pytest.mark.asyncio
async def test_send_pending_alerts_no_recipients(alert_agent, mock_state_manager):
    """Test send_pending_alerts when no recipients"""
    mock_state_manager.get_alert_recipients.return_value = []
    alert_agent.create_alert("test", "Test", "Message")

    result = await alert_agent.send_pending_alerts()

    assert result["processed"] == 0
    assert result["sent"] == 0
    assert result["skipped"] == 1


@pytest.mark.asyncio
async def test_send_pending_alerts_duplicate_skip(alert_agent):
    """Test send_pending_alerts skips duplicates"""
    alert = alert_agent.create_alert("test", "Test", "Message")
    alert_agent._sent_alert_ids.add(alert.id)  # Mark as already sent

    result = await alert_agent.send_pending_alerts()

    assert result["processed"] == 0
    assert result["skipped"] == 1


@pytest.mark.asyncio
async def test_send_pending_alerts_send_failure(alert_agent, mock_email_sender):
    """Test send_pending_alerts handles send failure"""
    mock_email_sender.send_alert.return_value = SendResult(
        success=False, sent_to=[], failed=["user@test.com"], message="Failed"
    )

    alert_agent.create_alert("test", "Test", "Message")
    result = await alert_agent.send_pending_alerts()

    assert result["processed"] == 1
    assert result["sent"] == 0
    assert result["failed"] == 1
    assert alert_agent._stats["emails_failed"] == 1


@pytest.mark.asyncio
async def test_send_pending_alerts_exception(alert_agent, mock_email_sender):
    """Test send_pending_alerts handles exceptions"""
    mock_email_sender.send_alert.side_effect = Exception("Email error")

    alert_agent.create_alert("test", "Test", "Message")
    result = await alert_agent.send_pending_alerts()

    assert result["processed"] == 1
    assert result["failed"] == 1


@pytest.mark.asyncio
async def test_send_pending_alerts_multiple(alert_agent, mock_state_manager):
    """Test sending multiple pending alerts"""
    alert_agent.create_alert("test1", "Test 1", "Message 1")
    alert_agent.create_alert("test2", "Test 2", "Message 2")
    alert_agent.create_alert("test3", "Test 3", "Message 3")

    result = await alert_agent.send_pending_alerts()

    assert result["processed"] == 3
    assert result["sent"] == 3
    assert len(alert_agent._pending_alerts) == 0


# =============================================================================
# Test _send_alert_email
# =============================================================================


@pytest.mark.asyncio
async def test_send_alert_email_rank_change(alert_agent, mock_email_sender):
    """Test _send_alert_email for rank_change type"""
    alert = Alert(
        id="test",
        type="rank_change",
        priority=AlertPriority.HIGH,
        title="Rank Change",
        message="Test",
        data={
            "product_name": "Product A",
            "brand": "Brand A",
            "previous_rank": 10,
            "current_rank": 20,
        },
    )

    result = await alert_agent._send_alert_email(alert, ["user@test.com"])

    mock_email_sender.send_rank_change_alert.assert_called_once_with(
        recipients=["user@test.com"],
        product_name="Product A",
        brand="Brand A",
        previous_rank=10,
        current_rank=20,
    )
    assert result.success is True


@pytest.mark.asyncio
async def test_send_alert_email_error(alert_agent, mock_email_sender):
    """Test _send_alert_email for error type"""
    alert = Alert(
        id="test",
        type="error",
        priority=AlertPriority.CRITICAL,
        title="Error",
        message="Error occurred",
        data={"location": "crawler"},
    )

    result = await alert_agent._send_alert_email(alert, ["user@test.com"])

    mock_email_sender.send_error_alert.assert_called_once_with(
        recipients=["user@test.com"],
        error_message="Error occurred",
        location="crawler",
    )
    assert result.success is True


@pytest.mark.asyncio
async def test_send_alert_email_generic(alert_agent, mock_email_sender):
    """Test _send_alert_email for generic alert type"""
    alert = Alert(
        id="test",
        type="important_insight",
        priority=AlertPriority.NORMAL,
        title="Insight",
        message="Important insight",
        data={"action_items": ["action1", "action2"]},
    )

    result = await alert_agent._send_alert_email(alert, ["user@test.com"])

    mock_email_sender.send_alert.assert_called_once_with(
        alert_type="important_insight",
        subject="Insight",
        content={
            "insight": "Important insight",
            "action_items": ["action1", "action2"],
        },
        recipients=["user@test.com"],
    )
    assert result.success is True


@pytest.mark.asyncio
async def test_send_alert_email_missing_data_fields(alert_agent, mock_email_sender):
    """Test _send_alert_email handles missing data fields"""
    alert = Alert(
        id="test",
        type="rank_change",
        priority=AlertPriority.HIGH,
        title="Test",
        message="Test",
        data={},  # Empty data
    )

    result = await alert_agent._send_alert_email(alert, ["user@test.com"])

    mock_email_sender.send_rank_change_alert.assert_called_once_with(
        recipients=["user@test.com"],
        product_name="Unknown",
        brand="Unknown",
        previous_rank=0,
        current_rank=0,
    )


# =============================================================================
# Test event handlers
# =============================================================================


@pytest.mark.asyncio
async def test_on_crawl_complete(alert_agent, mock_email_sender):
    """Test on_crawl_complete event handler"""
    data = {"total_products": 500, "categories": 5}

    await alert_agent.on_crawl_complete(data)

    assert len(alert_agent._alerts) == 1
    alert = alert_agent._alerts[0]
    assert alert.type == "crawl_complete"
    assert alert.priority == AlertPriority.LOW
    assert "크롤링 완료" in alert.title
    assert "500개 제품" in alert.message
    assert alert.data == data

    # Should trigger send_pending_alerts
    mock_email_sender.send_alert.assert_called_once()


@pytest.mark.asyncio
async def test_on_crawl_failed(alert_agent, mock_email_sender):
    """Test on_crawl_failed event handler"""
    data = {"error": "Connection timeout"}

    await alert_agent.on_crawl_failed(data)

    assert len(alert_agent._alerts) == 1
    alert = alert_agent._alerts[0]
    assert alert.type == "error"
    assert alert.priority == AlertPriority.CRITICAL
    assert "크롤링 실패" in alert.title
    assert alert.message == "Connection timeout"
    assert alert.data["location"] == "crawler"

    mock_email_sender.send_error_alert.assert_called_once()


@pytest.mark.asyncio
async def test_on_error(alert_agent, mock_email_sender):
    """Test on_error event handler"""
    data = {"error": "Database error", "location": "storage"}

    await alert_agent.on_error(data)

    assert len(alert_agent._alerts) == 1
    alert = alert_agent._alerts[0]
    assert alert.type == "error"
    assert alert.priority == AlertPriority.CRITICAL
    assert "시스템 에러" in alert.title
    assert alert.message == "Database error"
    assert alert.data == data

    mock_email_sender.send_error_alert.assert_called_once()


@pytest.mark.asyncio
async def test_on_error_unknown(alert_agent):
    """Test on_error with no error message"""
    data = {}

    await alert_agent.on_error(data)

    alert = alert_agent._alerts[0]
    assert alert.message == "알 수 없는 오류"


# =============================================================================
# Test send_daily_summary
# =============================================================================


@pytest.mark.asyncio
async def test_send_daily_summary_success(alert_agent, mock_state_manager, mock_email_sender):
    """Test successful daily summary sending"""
    mock_state_manager.get_alert_recipients.return_value = ["user1@test.com", "user2@test.com"]

    result = await alert_agent.send_daily_summary(
        highlights=["Highlight 1", "Highlight 2"],
        avg_rank=15.5,
        sos=0.25,
        alert_count=3,
        action_items=["Action 1", "Action 2"],
    )

    assert result.success is True
    assert result.sent_to == ["user@test.com"]
    mock_email_sender.send_daily_summary.assert_called_once_with(
        recipients=["user1@test.com", "user2@test.com"],
        highlights=["Highlight 1", "Highlight 2"],
        avg_rank=15.5,
        sos=0.25,
        alert_count=3,
        action_items=["Action 1", "Action 2"],
    )


@pytest.mark.asyncio
async def test_send_daily_summary_no_recipients(alert_agent, mock_state_manager):
    """Test daily summary when no recipients"""
    mock_state_manager.get_alert_recipients.return_value = []

    result = await alert_agent.send_daily_summary(
        highlights=[],
        avg_rank=10.0,
        sos=0.2,
        alert_count=0,
        action_items=[],
    )

    assert result.success is True
    assert result.sent_to == []
    assert "일일 요약 수신자 없음" in result.message


# =============================================================================
# Test get_alerts
# =============================================================================


def test_get_alerts_all(alert_agent):
    """Test get_alerts returns all alerts"""
    alert_agent.create_alert("test1", "Title 1", "Message 1")
    alert_agent.create_alert("test2", "Title 2", "Message 2")
    alert_agent.create_alert("test3", "Title 3", "Message 3")

    alerts = alert_agent.get_alerts()

    assert len(alerts) == 3
    # Should be sorted by created_at desc (newest first)
    assert alerts[0]["type"] == "test3"
    assert alerts[1]["type"] == "test2"
    assert alerts[2]["type"] == "test1"


def test_get_alerts_with_type_filter(alert_agent):
    """Test get_alerts with type filter"""
    alert_agent.create_alert("error", "Error 1", "Message 1")
    alert_agent.create_alert("rank_change", "Rank 1", "Message 2")
    alert_agent.create_alert("error", "Error 2", "Message 3")

    alerts = alert_agent.get_alerts(alert_type="error")

    assert len(alerts) == 2
    assert all(a["type"] == "error" for a in alerts)


def test_get_alerts_with_limit(alert_agent):
    """Test get_alerts respects limit"""
    for i in range(10):
        alert_agent.create_alert(f"test{i}", f"Title {i}", f"Message {i}")

    alerts = alert_agent.get_alerts(limit=5)

    assert len(alerts) == 5


def test_get_alerts_empty(alert_agent):
    """Test get_alerts when no alerts"""
    alerts = alert_agent.get_alerts()

    assert alerts == []


def test_get_alerts_sorted_by_time(alert_agent):
    """Test get_alerts returns alerts sorted by created_at descending"""
    # Create alerts with small delays to ensure different timestamps
    import time

    alert1 = alert_agent.create_alert("test1", "First", "Message 1")
    time.sleep(0.01)
    alert2 = alert_agent.create_alert("test2", "Second", "Message 2")
    time.sleep(0.01)
    alert3 = alert_agent.create_alert("test3", "Third", "Message 3")

    alerts = alert_agent.get_alerts()

    # Most recent first
    assert alerts[0]["id"] == alert3.id
    assert alerts[1]["id"] == alert2.id
    assert alerts[2]["id"] == alert1.id


# =============================================================================
# Test get_pending_count
# =============================================================================


def test_get_pending_count_initial(alert_agent):
    """Test get_pending_count initial state"""
    assert alert_agent.get_pending_count() == 0


def test_get_pending_count_after_create(alert_agent):
    """Test get_pending_count after creating alerts"""
    alert_agent.create_alert("test1", "Title 1", "Message 1")
    assert alert_agent.get_pending_count() == 1

    alert_agent.create_alert("test2", "Title 2", "Message 2")
    assert alert_agent.get_pending_count() == 2


@pytest.mark.asyncio
async def test_get_pending_count_after_send(alert_agent):
    """Test get_pending_count after sending alerts"""
    alert_agent.create_alert("test1", "Title 1", "Message 1")
    alert_agent.create_alert("test2", "Title 2", "Message 2")
    assert alert_agent.get_pending_count() == 2

    await alert_agent.send_pending_alerts()
    assert alert_agent.get_pending_count() == 0


# =============================================================================
# Test get_stats
# =============================================================================


def test_get_stats_initial(alert_agent):
    """Test get_stats initial state"""
    stats = alert_agent.get_stats()

    assert stats["total_alerts"] == 0
    assert stats["emails_sent"] == 0
    assert stats["emails_failed"] == 0
    assert stats["pending"] == 0
    assert stats["email_enabled"] is True


def test_get_stats_after_alerts(alert_agent):
    """Test get_stats after creating alerts"""
    alert_agent.create_alert("test1", "Title 1", "Message 1")
    alert_agent.create_alert("test2", "Title 2", "Message 2")

    stats = alert_agent.get_stats()

    assert stats["total_alerts"] == 2
    assert stats["pending"] == 2


@pytest.mark.asyncio
async def test_get_stats_after_send(alert_agent, mock_state_manager, mock_email_sender):
    """Test get_stats after sending alerts"""
    mock_state_manager.get_alert_recipients.return_value = ["user1@test.com", "user2@test.com"]
    # Update mock to return 2 recipients in sent_to
    mock_email_sender.send_alert.return_value = SendResult(
        success=True, sent_to=["user1@test.com", "user2@test.com"], failed=[], message="OK"
    )

    alert_agent.create_alert("test1", "Title 1", "Message 1")
    await alert_agent.send_pending_alerts()

    stats = alert_agent.get_stats()

    assert stats["total_alerts"] == 1
    assert stats["emails_sent"] == 2  # 2 recipients
    assert stats["pending"] == 0


def test_get_stats_email_disabled(alert_agent, mock_email_sender):
    """Test get_stats when email is disabled"""
    mock_email_sender.is_enabled.return_value = False

    stats = alert_agent.get_stats()

    assert stats["email_enabled"] is False


# =============================================================================
# Test clear_old_alerts
# =============================================================================


def test_clear_old_alerts_no_old_alerts(alert_agent):
    """Test clear_old_alerts when no old alerts"""
    alert_agent.create_alert("test1", "Title 1", "Message 1")

    removed = alert_agent.clear_old_alerts(hours=24)

    assert removed == 0
    assert len(alert_agent._alerts) == 1


def test_clear_old_alerts_removes_old(alert_agent):
    """Test clear_old_alerts removes old alerts"""
    # Create old alert
    old_alert = Alert(
        id="old_001",
        type="test",
        priority=AlertPriority.NORMAL,
        title="Old Alert",
        message="Old",
        created_at=datetime.now() - timedelta(hours=25),
    )
    alert_agent._alerts.append(old_alert)
    alert_agent._sent_alert_ids.add(old_alert.id)

    # Create recent alert
    alert_agent.create_alert("test", "Recent", "Recent message")

    removed = alert_agent.clear_old_alerts(hours=24)

    assert removed == 1
    assert len(alert_agent._alerts) == 1
    assert alert_agent._alerts[0].title == "Recent"
    assert old_alert.id not in alert_agent._sent_alert_ids


def test_clear_old_alerts_custom_hours(alert_agent):
    """Test clear_old_alerts with custom hours threshold"""
    # Create alert that's 10 hours old
    old_alert = Alert(
        id="old_001",
        type="test",
        priority=AlertPriority.NORMAL,
        title="Old Alert",
        message="Old",
        created_at=datetime.now() - timedelta(hours=10),
    )
    alert_agent._alerts.append(old_alert)

    # Clear alerts older than 5 hours
    removed = alert_agent.clear_old_alerts(hours=5)

    assert removed == 1
    assert len(alert_agent._alerts) == 0


def test_clear_old_alerts_multiple(alert_agent):
    """Test clear_old_alerts removes multiple old alerts"""
    # Create 3 old alerts
    for i in range(3):
        old_alert = Alert(
            id=f"old_{i:03d}",
            type="test",
            priority=AlertPriority.NORMAL,
            title=f"Old {i}",
            message=f"Old message {i}",
            created_at=datetime.now() - timedelta(hours=30),
        )
        alert_agent._alerts.append(old_alert)
        alert_agent._sent_alert_ids.add(old_alert.id)

    # Create 2 recent alerts
    alert_agent.create_alert("test1", "Recent 1", "Message 1")
    alert_agent.create_alert("test2", "Recent 2", "Message 2")

    removed = alert_agent.clear_old_alerts(hours=24)

    assert removed == 3
    assert len(alert_agent._alerts) == 2
    assert len(alert_agent._sent_alert_ids) == 0


# =============================================================================
# Edge Cases & Error Handling
# =============================================================================


@pytest.mark.asyncio
async def test_process_metrics_with_boundary_rank_change(alert_agent):
    """Test process_metrics with boundary rank changes (exactly 10 or -10)"""
    metrics_data = {
        "products": [
            {
                "name": "Product A",
                "brand": "Brand A",
                "previous_rank": 10,
                "current_rank": 20,
                "rank_change": 10,  # Exactly 10 (should trigger)
            },
            {
                "name": "Product B",
                "brand": "Brand B",
                "previous_rank": 20,
                "current_rank": 10,
                "rank_change": -10,  # Exactly -10 (should trigger)
            },
        ]
    }

    alerts = await alert_agent.process_metrics(metrics_data)

    # Should create both alerts (boundaries are inclusive)
    assert len(alerts) >= 2


@pytest.mark.asyncio
async def test_process_metrics_top10_boundary(alert_agent):
    """Test process_metrics with Top 10 boundary (rank 10)"""
    metrics_data = {
        "products": [
            {
                "name": "Product A",
                "brand": "Brand A",
                "previous_rank": 11,
                "current_rank": 10,  # Exactly 10 (should trigger)
                "rank_change": -1,
            }
        ]
    }

    alerts = await alert_agent.process_metrics(metrics_data)

    assert len(alerts) == 1
    assert alerts[0].type == "important_insight"


@pytest.mark.asyncio
async def test_send_pending_alerts_partial_failure(alert_agent, mock_email_sender):
    """Test send_pending_alerts with partial send success"""
    mock_email_sender.send_alert.return_value = SendResult(
        success=True,
        sent_to=["user1@test.com"],
        failed=["user2@test.com"],
        message="Partial success",
    )

    alert_agent.create_alert("test", "Test", "Message")
    result = await alert_agent.send_pending_alerts()

    assert result["sent"] == 1
    assert alert_agent._stats["emails_sent"] == 1


def test_alert_agent_isolation(mock_state_manager, mock_rules_engine, mock_email_sender):
    """Test that multiple AlertAgent instances are isolated"""
    agent1 = AlertAgent(
        state_manager=mock_state_manager,
        rules_engine=mock_rules_engine,
        email_sender=mock_email_sender,
    )
    agent2 = AlertAgent(
        state_manager=mock_state_manager,
        rules_engine=mock_rules_engine,
        email_sender=mock_email_sender,
    )

    agent1.create_alert("test1", "Title 1", "Message 1")

    assert len(agent1._alerts) == 1
    assert len(agent2._alerts) == 0


@pytest.mark.asyncio
async def test_send_pending_alerts_updates_alert_state(alert_agent):
    """Test send_pending_alerts updates alert sent state"""
    alert = alert_agent.create_alert("test", "Test", "Message")

    assert alert.sent is False
    assert alert.sent_to == []

    await alert_agent.send_pending_alerts()

    assert alert.sent is True
    assert alert.sent_to == ["user@test.com"]
