"""Session data encryption utilities.

Uses Fernet symmetric encryption when cryptography is available.
Falls back to base64 encoding (NOT secure) with a warning if unavailable.
"""

import base64
import json
import logging
import os

logger = logging.getLogger(__name__)

_ENCRYPTION_KEY = os.environ.get("SESSION_ENCRYPTION_KEY", "")

try:
    from cryptography.fernet import Fernet

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.info("cryptography not installed; session encryption disabled")


def _get_fernet() -> "Fernet | None":
    """Get or create Fernet instance from environment key."""
    if not CRYPTO_AVAILABLE or not _ENCRYPTION_KEY:
        return None
    try:
        return Fernet(_ENCRYPTION_KEY.encode())
    except Exception:
        logger.warning("Invalid SESSION_ENCRYPTION_KEY; encryption disabled")
        return None


def encrypt_session_data(data: dict) -> str:
    """Encrypt session data dict to string."""
    raw = json.dumps(data, ensure_ascii=False)
    fernet = _get_fernet()
    if fernet:
        return fernet.encrypt(raw.encode()).decode()
    return base64.b64encode(raw.encode()).decode()


def decrypt_session_data(encrypted: str) -> dict:
    """Decrypt session data string to dict."""
    fernet = _get_fernet()
    if fernet:
        try:
            raw = fernet.decrypt(encrypted.encode()).decode()
            return json.loads(raw)
        except Exception:
            logger.warning("Session decryption failed; attempting base64 fallback")
    try:
        raw = base64.b64decode(encrypted.encode()).decode()
        return json.loads(raw)
    except Exception:
        logger.warning("Session data decode failed; returning empty dict")
        return {}
