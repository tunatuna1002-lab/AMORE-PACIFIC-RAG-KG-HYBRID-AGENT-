"""
TikTok Collector 단위 테스트 - mask_proxy_url
"""

from src.tools.collectors.tiktok_collector import mask_proxy_url


class TestMaskProxyUrl:
    def test_masks_credentials(self):
        """Credentials in proxy URL should be masked."""
        url = "http://user:password@proxy.example.com:8080"  # pragma: allowlist secret
        result = mask_proxy_url(url)
        assert "user" not in result
        assert "password" not in result
        assert "***:***@" in result
        assert "proxy.example.com" in result
        assert "8080" in result

    def test_no_credentials(self):
        """URL without credentials should be returned as-is."""
        url = "http://proxy.example.com:8080"
        result = mask_proxy_url(url)
        assert result == url

    def test_credentials_without_port(self):
        """Proxy URL with credentials but no port."""
        url = "http://admin:secret@proxy.example.com"  # pragma: allowlist secret
        result = mask_proxy_url(url)
        assert "admin" not in result
        assert "secret" not in result
        assert "***:***@proxy.example.com" in result

    def test_socks5_proxy(self):
        """SOCKS5 proxy with credentials."""
        url = "socks5://myuser:mypass@socks.proxy.io:1080"  # pragma: allowlist secret
        result = mask_proxy_url(url)
        assert "myuser" not in result
        assert "mypass" not in result
        assert "socks5://" in result
        assert "socks.proxy.io" in result

    def test_empty_string(self):
        """Empty string should be returned as-is."""
        assert mask_proxy_url("") == ""

    def test_plain_host(self):
        """Plain hostname without scheme should be returned as-is."""
        url = "proxy.example.com"
        result = mask_proxy_url(url)
        assert result == url
