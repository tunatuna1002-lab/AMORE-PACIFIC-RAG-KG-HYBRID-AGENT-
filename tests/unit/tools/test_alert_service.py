"""
AlertService 단위 테스트

테스트 대상: src/tools/notifications/alert_service.py
Coverage target: 60%+
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.notifications.alert_service import (
    AlertConfig,
    AlertService,
    get_alert_service,
)


class TestAlertConfig:
    """AlertConfig 데이터클래스 테스트"""

    def test_default_values(self):
        """기본값으로 초기화"""
        config = AlertConfig()

        assert config.slack_webhook_url is None
        assert config.slack_channel == "#deals-alert"
        assert config.smtp_host is None
        assert config.smtp_port == 587
        assert config.smtp_user is None
        assert config.smtp_password is None
        assert config.email_recipients == []
        assert config.email_from == "deals-alert@amore.com"
        assert config.min_discount_percent == 20.0
        assert config.alert_brands == []

    def test_custom_values(self):
        """커스텀 값으로 초기화"""
        config = AlertConfig(
            slack_webhook_url="https://hooks.slack.com/test",
            slack_channel="#test-channel",
            smtp_host="smtp.test.com",
            smtp_port=465,
            smtp_user="user@test.com",
            smtp_password="password123",  # pragma: allowlist secret
            email_recipients=["a@test.com", "b@test.com"],
            email_from="from@test.com",
            min_discount_percent=30.0,
            alert_brands=["COSRX", "ANUA"],
        )

        assert config.slack_webhook_url == "https://hooks.slack.com/test"
        assert config.slack_channel == "#test-channel"
        assert config.smtp_host == "smtp.test.com"
        assert config.smtp_port == 465
        assert config.email_recipients == ["a@test.com", "b@test.com"]
        assert config.alert_brands == ["COSRX", "ANUA"]

    def test_post_init_none_recipients(self):
        """email_recipients가 None이면 빈 리스트로 초기화"""
        config = AlertConfig(email_recipients=None)
        assert config.email_recipients == []

    def test_post_init_none_brands(self):
        """alert_brands가 None이면 빈 리스트로 초기화"""
        config = AlertConfig(alert_brands=None)
        assert config.alert_brands == []


class TestAlertServiceInit:
    """AlertService 초기화 테스트"""

    def test_init_with_config(self):
        """AlertConfig 지정 시 해당 설정 사용"""
        config = AlertConfig(
            slack_webhook_url="https://hooks.slack.com/test",
            smtp_host="smtp.test.com",
            smtp_user="user@test.com",
            email_recipients=["a@test.com"],
        )
        service = AlertService(config=config)

        assert service._slack_enabled is True
        assert service._email_enabled is True

    def test_init_without_slack(self):
        """Slack 미설정 시 비활성화"""
        config = AlertConfig(slack_webhook_url=None)
        service = AlertService(config=config)

        assert service._slack_enabled is False

    def test_init_without_email(self):
        """Email 미설정 시 비활성화"""
        config = AlertConfig(smtp_host=None, smtp_user=None, email_recipients=[])
        service = AlertService(config=config)

        assert service._email_enabled is False

    def test_init_email_requires_all_fields(self):
        """Email은 host, user, recipients 모두 필요"""
        # host만 있는 경우
        config = AlertConfig(smtp_host="smtp.test.com", smtp_user=None, email_recipients=[])
        service = AlertService(config=config)
        assert service._email_enabled is False

        # host + user만 있는 경우
        config = AlertConfig(
            smtp_host="smtp.test.com",
            smtp_user="user@test.com",
            email_recipients=[],
        )
        service = AlertService(config=config)
        assert service._email_enabled is False

    @patch.dict(
        "os.environ",
        {
            "SLACK_WEBHOOK_URL": "https://hooks.slack.com/env",
            "SLACK_CHANNEL": "#env-channel",
            "SMTP_HOST": "smtp.env.com",
            "SMTP_PORT": "465",
            "SMTP_USER": "envuser",
            "SMTP_PASSWORD": "envpass",  # pragma: allowlist secret
            "ALERT_EMAIL_RECIPIENTS": "a@env.com, b@env.com",
            "ALERT_EMAIL_FROM": "from@env.com",
            "ALERT_MIN_DISCOUNT": "25.0",
            "ALERT_BRANDS": "COSRX, ANUA",
        },
        clear=False,
    )
    def test_load_config_from_env(self):
        """환경변수에서 설정 로드"""
        service = AlertService()

        assert service.config.slack_webhook_url == "https://hooks.slack.com/env"
        assert service.config.slack_channel == "#env-channel"
        assert service.config.smtp_host == "smtp.env.com"
        assert service.config.smtp_port == 465
        assert service.config.smtp_user == "envuser"
        assert service.config.smtp_password == "envpass"  # pragma: allowlist secret
        assert service.config.email_recipients == ["a@env.com", "b@env.com"]
        assert service.config.email_from == "from@env.com"
        assert service.config.min_discount_percent == 25.0
        assert service.config.alert_brands == ["COSRX", "ANUA"]

    @patch.dict("os.environ", {}, clear=True)
    def test_load_config_from_env_empty(self):
        """환경변수가 없을 때 기본값 사용"""
        service = AlertService()

        assert service.config.slack_webhook_url is None
        assert service.config.smtp_host is None
        assert service.config.email_recipients == []
        assert service.config.min_discount_percent == 20.0


class TestAlertServiceCheckDeal:
    """_check_deal_for_alert 메서드 테스트"""

    def _make_service(self, alert_brands=None):
        config = AlertConfig(alert_brands=alert_brands)
        return AlertService(config=config)

    def test_competitor_lightning_deal(self):
        """경쟁사 Lightning Deal 감지"""
        service = self._make_service()
        deal = {
            "brand": "COSRX",
            "discount_percent": 15,
            "deal_type": "lightning",
            "product_name": "Advanced Snail Mucin",
        }

        alert = service._check_deal_for_alert(deal)

        assert alert is not None
        assert alert["alert_type"] == "lightning_deal"
        assert "COSRX" in alert["alert_message"]
        assert "Lightning Deal" in alert["alert_message"]

    def test_competitor_big_discount(self):
        """경쟁사 30% 이상 할인 감지"""
        service = self._make_service()
        deal = {
            "brand": "Beauty of Joseon",
            "discount_percent": 35,
            "deal_type": "normal",
        }

        alert = service._check_deal_for_alert(deal)

        assert alert is not None
        assert alert["alert_type"] == "big_discount"
        assert "35%" in alert["alert_message"]

    def test_competitor_deal_of_day(self):
        """경쟁사 Deal of the Day 감지"""
        service = self._make_service()
        deal = {
            "brand": "SKIN1004",
            "discount_percent": 15,
            "deal_type": "deal_of_day",
        }

        alert = service._check_deal_for_alert(deal)

        assert alert is not None
        assert alert["alert_type"] == "deal_of_day"

    def test_competitor_promo_above_threshold(self):
        """경쟁사 최소 할인율 이상 프로모션"""
        service = self._make_service()
        deal = {
            "brand": "ANUA",
            "discount_percent": 22,
            "deal_type": "normal",
        }

        alert = service._check_deal_for_alert(deal)

        assert alert is not None
        assert alert["alert_type"] == "competitor_promo"

    def test_non_competitor_ignored(self):
        """경쟁사가 아닌 브랜드는 무시"""
        service = self._make_service()
        deal = {
            "brand": "Unknown Brand XYZ",
            "discount_percent": 50,
            "deal_type": "lightning",
        }

        alert = service._check_deal_for_alert(deal)

        assert alert is None

    def test_below_threshold_ignored(self):
        """할인율이 기준 미만이면 무시"""
        service = self._make_service()
        deal = {
            "brand": "COSRX",
            "discount_percent": 10,
            "deal_type": "normal",
        }

        alert = service._check_deal_for_alert(deal)

        assert alert is None

    def test_brand_filter(self):
        """특정 브랜드만 모니터링"""
        service = self._make_service(alert_brands=["COSRX"])
        deal_cosrx = {
            "brand": "COSRX",
            "discount_percent": 25,
            "deal_type": "normal",
        }
        deal_anua = {
            "brand": "ANUA",
            "discount_percent": 25,
            "deal_type": "normal",
        }

        assert service._check_deal_for_alert(deal_cosrx) is not None
        assert service._check_deal_for_alert(deal_anua) is None

    def test_case_insensitive_brand_matching(self):
        """대소문자 구분 없이 브랜드 매칭"""
        service = self._make_service()
        deal = {
            "brand": "cosrx",
            "discount_percent": 35,
            "deal_type": "normal",
        }

        alert = service._check_deal_for_alert(deal)
        assert alert is not None

    def test_alert_contains_deal_fields(self):
        """알림에 딜 정보 필드가 포함"""
        service = self._make_service()
        deal = {
            "brand": "COSRX",
            "discount_percent": 25,
            "deal_type": "lightning",
            "asin": "B0TEST",
            "product_name": "Test Product",
            "deal_price": 15.99,
            "original_price": 25.99,
            "time_remaining": "2h 30m",
            "claimed_percent": 45,
            "product_url": "https://amazon.com/dp/B0TEST",
        }

        alert = service._check_deal_for_alert(deal)

        assert alert is not None
        assert alert["brand"] == "COSRX"
        assert alert["asin"] == "B0TEST"
        assert alert["product_name"] == "Test Product"
        assert alert["deal_price"] == 15.99
        assert alert["original_price"] == 25.99
        assert alert["time_remaining"] == "2h 30m"
        assert alert["claimed_percent"] == 45
        assert alert["product_url"] == "https://amazon.com/dp/B0TEST"
        assert "alert_datetime" in alert

    def test_none_discount_percent(self):
        """discount_percent가 None인 경우"""
        service = self._make_service()
        deal = {
            "brand": "COSRX",
            "discount_percent": None,
            "deal_type": "normal",
        }

        alert = service._check_deal_for_alert(deal)
        assert alert is None


class TestAlertServiceProcessDeals:
    """process_deals_for_alerts 메서드 테스트"""

    @pytest.mark.asyncio
    async def test_process_deals_with_alerts(self):
        """알림 대상 딜이 있는 경우"""
        config = AlertConfig()
        service = AlertService(config=config)
        service._send_alerts_batch = AsyncMock()

        deals = [
            {"brand": "COSRX", "discount_percent": 35, "deal_type": "normal"},
            {"brand": "Unknown", "discount_percent": 10, "deal_type": "normal"},
            {"brand": "ANUA", "discount_percent": 25, "deal_type": "normal"},
        ]

        alerts = await service.process_deals_for_alerts(deals)

        assert len(alerts) == 2
        service._send_alerts_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_deals_no_alerts(self):
        """알림 대상 딜이 없는 경우"""
        config = AlertConfig()
        service = AlertService(config=config)
        service._send_alerts_batch = AsyncMock()

        deals = [
            {"brand": "Unknown", "discount_percent": 5, "deal_type": "normal"},
        ]

        alerts = await service.process_deals_for_alerts(deals)

        assert len(alerts) == 0
        service._send_alerts_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_empty_deals(self):
        """빈 딜 리스트"""
        config = AlertConfig()
        service = AlertService(config=config)
        service._send_alerts_batch = AsyncMock()

        alerts = await service.process_deals_for_alerts([])

        assert len(alerts) == 0
        service._send_alerts_batch.assert_not_called()


class TestAlertServiceSendAlertsBatch:
    """_send_alerts_batch 메서드 테스트"""

    @pytest.mark.asyncio
    async def test_send_batch_empty(self):
        """빈 알림 리스트 시 아무것도 안함"""
        config = AlertConfig()
        service = AlertService(config=config)

        await service._send_alerts_batch([])
        # 에러 없이 완료

    @pytest.mark.asyncio
    async def test_send_batch_slack_enabled(self):
        """Slack이 활성화된 경우"""
        config = AlertConfig(slack_webhook_url="https://hooks.slack.com/test")
        service = AlertService(config=config)
        service._send_slack_batch = AsyncMock(return_value=True)

        alerts = [{"alert_type": "big_discount", "brand": "COSRX"}]
        await service._send_alerts_batch(alerts)

        service._send_slack_batch.assert_called_once_with(alerts)

    @pytest.mark.asyncio
    async def test_send_batch_email_enabled(self):
        """Email이 활성화된 경우"""
        config = AlertConfig(
            smtp_host="smtp.test.com",
            smtp_user="user@test.com",
            email_recipients=["a@test.com"],
        )
        service = AlertService(config=config)
        service._send_email_batch = AsyncMock(return_value=True)

        alerts = [{"alert_type": "big_discount", "brand": "COSRX"}]
        await service._send_alerts_batch(alerts)

        service._send_email_batch.assert_called_once_with(alerts)

    @pytest.mark.asyncio
    async def test_send_batch_slack_error_handled(self):
        """Slack 전송 실패 시 에러 로그만 남김"""
        config = AlertConfig(slack_webhook_url="https://hooks.slack.com/test")
        service = AlertService(config=config)
        service._send_slack_batch = AsyncMock(side_effect=Exception("Slack error"))

        alerts = [{"alert_type": "big_discount", "brand": "COSRX"}]
        # 에러 발생해도 예외를 던지지 않아야 함
        await service._send_alerts_batch(alerts)

    @pytest.mark.asyncio
    async def test_send_batch_email_error_handled(self):
        """Email 전송 실패 시 에러 로그만 남김"""
        config = AlertConfig(
            smtp_host="smtp.test.com",
            smtp_user="user@test.com",
            email_recipients=["a@test.com"],
        )
        service = AlertService(config=config)
        service._send_email_batch = AsyncMock(side_effect=Exception("Email error"))

        alerts = [{"alert_type": "big_discount", "brand": "COSRX"}]
        await service._send_alerts_batch(alerts)


class TestAlertServiceSlack:
    """Slack 전송 테스트"""

    @pytest.mark.asyncio
    async def test_slack_disabled(self):
        """Slack 미설정 시 False 반환"""
        config = AlertConfig(slack_webhook_url=None)
        service = AlertService(config=config)

        result = await service._send_slack_batch([])
        assert result is False

    @pytest.mark.asyncio
    async def test_slack_send_success(self):
        """Slack 전송 성공"""
        config = AlertConfig(slack_webhook_url="https://hooks.slack.com/test")
        service = AlertService(config=config)

        alerts = [
            {
                "alert_type": "big_discount",
                "brand": "COSRX",
                "discount_percent": 35,
                "product_name": "Test Product Name",
                "product_url": "https://amazon.com/dp/B0TEST",
                "deal_price": 15.99,
                "time_remaining": "2h 30m",
                "claimed_percent": 45,
            }
        ]

        mock_response = AsyncMock()
        mock_response.status = 200

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=None),
            )
        )

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await service._send_slack_batch(alerts)

        assert result is True

    @pytest.mark.asyncio
    async def test_slack_truncates_long_product_name(self):
        """긴 제품명 truncation"""
        config = AlertConfig(slack_webhook_url="https://hooks.slack.com/test")
        service = AlertService(config=config)

        alerts = [
            {
                "alert_type": "big_discount",
                "brand": "COSRX",
                "discount_percent": 35,
                "product_name": "A" * 60,  # 50자 초과
                "product_url": "https://amazon.com/dp/B0TEST",
            }
        ]

        mock_response = AsyncMock()
        mock_response.status = 200

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=None),
            )
        )

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await service._send_slack_batch(alerts)

        assert result is True


class TestAlertServiceEmail:
    """Email 전송 테스트"""

    @pytest.mark.asyncio
    async def test_email_disabled(self):
        """Email 미설정 시 False 반환"""
        config = AlertConfig()
        service = AlertService(config=config)

        result = await service._send_email_batch([])
        assert result is False

    def test_build_email_html(self):
        """HTML 이메일 본문 생성"""
        config = AlertConfig()
        service = AlertService(config=config)

        alerts = [
            {
                "alert_type": "big_discount",
                "brand": "COSRX",
                "discount_percent": 35,
                "product_name": "Test Product",
                "deal_price": 15.99,
                "product_url": "https://amazon.com/dp/B0TEST",
                "asin": "B0TEST",
            }
        ]

        html = service._build_email_html(alerts)

        assert "COSRX" in html
        assert "35%" in html or "35" in html
        assert "Test Product" in html
        assert "경쟁사 할인 알림" in html

    def test_build_email_html_long_product_name(self):
        """긴 제품명 HTML에서 truncation"""
        config = AlertConfig()
        service = AlertService(config=config)

        alerts = [
            {
                "alert_type": "big_discount",
                "brand": "COSRX",
                "discount_percent": 35,
                "product_name": "A" * 50,  # 40자 초과
                "deal_price": 15.99,
                "asin": "B0TEST",
            }
        ]

        html = service._build_email_html(alerts)
        assert "..." in html

    def test_build_email_plain(self):
        """Plain text 이메일 본문 생성"""
        config = AlertConfig()
        service = AlertService(config=config)

        alerts = [
            {
                "alert_type": "big_discount",
                "brand": "COSRX",
                "discount_percent": 35,
                "product_name": "Test Product",
                "deal_price": 15.99,
                "product_url": "https://amazon.com/dp/B0TEST",
            }
        ]

        text = service._build_email_plain(alerts)

        assert "COSRX" in text
        assert "35%" in text
        assert "경쟁사 할인 알림" in text

    def test_build_email_plain_more_than_20(self):
        """20건 초과 시 '외 N건' 표시"""
        config = AlertConfig()
        service = AlertService(config=config)

        alerts = [
            {
                "alert_type": "big_discount",
                "brand": f"Brand{i}",
                "discount_percent": 30 + i,
                "product_name": f"Product {i}",
                "deal_price": 10.0 + i,
                "asin": f"B{i:04d}",
            }
            for i in range(25)
        ]

        text = service._build_email_plain(alerts)
        assert "외 5건" in text


class TestAlertServiceSingleAlert:
    """send_single_alert 메서드 테스트"""

    @pytest.mark.asyncio
    async def test_send_single_alert_both_channels(self):
        """단일 알림 양쪽 채널 전송"""
        config = AlertConfig(
            slack_webhook_url="https://hooks.slack.com/test",
            smtp_host="smtp.test.com",
            smtp_user="user@test.com",
            email_recipients=["a@test.com"],
        )
        service = AlertService(config=config)
        service._send_slack_batch = AsyncMock(return_value=True)
        service._send_email_batch = AsyncMock(return_value=True)

        alert = {"alert_type": "big_discount", "brand": "COSRX"}
        results = await service.send_single_alert(alert)

        assert results["slack"] is True
        assert results["email"] is True

    @pytest.mark.asyncio
    async def test_send_single_alert_no_channels(self):
        """채널 비활성 시 모두 False"""
        config = AlertConfig()
        service = AlertService(config=config)

        alert = {"alert_type": "big_discount", "brand": "COSRX"}
        results = await service.send_single_alert(alert)

        assert results["slack"] is False
        assert results["email"] is False


class TestAlertServiceGetStatus:
    """get_status 메서드 테스트"""

    def test_get_status_all_enabled(self):
        """모든 채널 활성 상태"""
        config = AlertConfig(
            slack_webhook_url="https://hooks.slack.com/test",
            smtp_host="smtp.test.com",
            smtp_user="user@test.com",
            email_recipients=["a@test.com", "b@test.com"],
            min_discount_percent=25.0,
            alert_brands=["COSRX"],
        )
        service = AlertService(config=config)

        status = service.get_status()

        assert status["slack_enabled"] is True
        assert status["email_enabled"] is True
        assert status["email_recipients"] == 2
        assert status["min_discount_threshold"] == 25.0
        assert status["monitored_brands"] == ["COSRX"]
        assert "competitor_brands" in status

    def test_get_status_disabled(self):
        """모든 채널 비활성 상태"""
        config = AlertConfig()
        service = AlertService(config=config)

        status = service.get_status()

        assert status["slack_enabled"] is False
        assert status["email_enabled"] is False
        assert status["email_recipients"] == 0
        assert status["monitored_brands"] == "ALL"


class TestGetAlertServiceSingleton:
    """get_alert_service 싱글톤 테스트"""

    def test_get_alert_service_returns_instance(self):
        """싱글톤 인스턴스 반환"""
        import src.tools.notifications.alert_service as module

        module._alert_service_instance = None

        with patch.dict("os.environ", {}, clear=True):
            service = get_alert_service()

        assert isinstance(service, AlertService)

        # 두 번째 호출에서 같은 인스턴스
        service2 = get_alert_service()
        assert service is service2

        # cleanup
        module._alert_service_instance = None
