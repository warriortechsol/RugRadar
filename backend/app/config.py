import os
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

@dataclass(frozen=True)
class Settings:
    # --- Ethereum ---
    ALCHEMY_ETH_HTTP_URL: Optional[str] = os.getenv("ALCHEMY_ETH_HTTP_URL")

    # --- Solana ---
    SOLANA_RPC_URLS_RAW: Optional[str] = os.getenv("SOLANA_RPC_URLS")
    SOLSCAN_API_TOKEN: Optional[str] = os.getenv("SOLSCAN_API_TOKEN")

    # --- API / CORS ---
    CORS_ALLOW_ORIGINS: str = os.getenv("CORS_ALLOW_ORIGINS", "*")

    # --- Tuning ---
    HTTP_TIMEOUT_SECONDS: int = int(os.getenv("HTTP_TIMEOUT_SECONDS", "20"))
    RETRY_MAX_ATTEMPTS: int = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
    RETRY_BASE_DELAY_MS: int = int(os.getenv("RETRY_BASE_DELAY_MS", "200"))

    @property
    def SOLANA_RPC_URLS(self) -> List[str]:
        raw = self.SOLANA_RPC_URLS_RAW or ""
        return [u.strip() for u in raw.split(",") if u.strip()]

settings = Settings()

def validate_settings(require_ethereum: bool = False) -> None:
    """
    Validate environment settings.
    If require_ethereum=True, ensure Ethereum settings are present.
    """
    if require_ethereum and not settings.ALCHEMY_ETH_HTTP_URL:
        raise RuntimeError("Alchemy ETH URL not configured")
