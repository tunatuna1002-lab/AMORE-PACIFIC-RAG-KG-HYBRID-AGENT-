"""Tests for session data encryption."""

from src.memory.session_crypto import decrypt_session_data, encrypt_session_data


class TestSessionCrypto:
    """Session encryption/decryption tests."""

    def test_encrypt_decrypt_roundtrip(self):
        data = {"user": "test", "count": 42}
        encrypted = encrypt_session_data(data)
        assert encrypted != str(data)
        result = decrypt_session_data(encrypted)
        assert result == data

    def test_empty_data(self):
        data = {}
        encrypted = encrypt_session_data(data)
        result = decrypt_session_data(encrypted)
        assert result == data

    def test_decrypt_invalid_returns_empty(self):
        result = decrypt_session_data("not-valid-data!!!")
        assert result == {}
