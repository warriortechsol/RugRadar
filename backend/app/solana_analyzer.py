import os
import re
import random
import time
import asyncio
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

import httpx
from fastapi import HTTPException
from pydantic import BaseModel
import redis.asyncio as redis
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Load env + validate settings
# -----------------------------------------------------------------------------
load_dotenv()
from app.config import validate_settings
validate_settings()

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
PORT = int(os.getenv("PORT", "8000"))
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "20"))
RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
RETRY_BASE_DELAY_MS = int(os.getenv("RETRY_BASE_DELAY_MS", "200"))

SOLANA_RPC_URLS = [
    url.strip()
    for url in os.getenv("SOLANA_RPC_URLS", "https://api.mainnet-beta.solana.com").split(",")
    if url.strip()
]
SOLSCAN_API_TOKEN = os.getenv("SOLSCAN_API_TOKEN", "").strip()

# Optional Redis (for burn tracking)
REDIS_URL = os.getenv("REDIS_URL", "").strip()
redis_client: Optional[redis.Redis] = None
if REDIS_URL:
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PLACEHOLDER_LOGO = os.getenv("PLACEHOLDER_LOGO", "/placeholder.png").strip()
TRANSIENT_STATUS = {429, 500, 502, 503, 504}
JSONRPC_RATE_LIMIT_CODES = {-32005, -32011}

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class Metadata(BaseModel):
    mint: str
    owner: Optional[str] = None
    token_amount: Optional[int] = None
    decimals: Optional[int] = None
    supply: Optional[int] = None
    name: Optional[str] = None
    symbol: Optional[str] = None
    logo_uri: Optional[str] = None

class RiskScore(BaseModel):
    score: int
    reasons: List[str]

class AnalyzeResult(BaseModel):
    address: str
    chain: str
    metadata: Metadata
    risk_score: RiskScore
    events: Optional[List[Dict[str, Any]]] = None
    holders: List[Dict[str, Any]] = []  # <-- always a list, defaults to empty



class TokenHolding(BaseModel):
    mint: str
    amount: int
    decimals: int
    ui_amount: float
    name: Optional[str] = None
    symbol: Optional[str] = None
    logo_uri: Optional[str] = None
# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def sanitize_address(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().split("#", 1)[0].split("?", 1)[0]
    if s.lower().endswith("pump"):
        s = s[: -len("pump")]
    return s.strip()

def is_solana_address(addr: str) -> bool:
    return bool(re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{32,44}", addr))

def _sym_from_mint(mint: str) -> str:
    return f"{mint[:4]}…{mint[-4:]}" if isinstance(mint, str) and len(mint) >= 8 else (mint or "UNK")

def _fill_metadata_defaults(h: TokenHolding) -> None:
    if not h.symbol or not str(h.symbol).strip():
        h.symbol = _sym_from_mint(h.mint)
    if not h.name or not str(h.name).strip():
        h.name = h.symbol

# -----------------------------------------------------------------------------
# HTTP client helpers
# -----------------------------------------------------------------------------
async def http_post_json(url: Union[str, List[str]], json: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    urls = [url] if isinstance(url, str) else list(url)
    delay = RETRY_BASE_DELAY_MS / 1000.0
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        target = urls[(attempt - 1) % len(urls)]
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
                r = await client.post(target, json=json, headers=headers)
                r.raise_for_status()
                return r.json()
        except httpx.HTTPStatusError as e:
            if attempt == RETRY_MAX_ATTEMPTS or (e.response is not None and e.response.status_code not in TRANSIENT_STATUS):
                raise
        except httpx.RequestError:
            if attempt == RETRY_MAX_ATTEMPTS:
                raise
        await asyncio.sleep(delay + random.uniform(0, delay * 0.25))
        delay *= 2

async def http_get_json(url: str, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> Dict[str, Any]:
    delay = RETRY_BASE_DELAY_MS / 1000.0
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout or HTTP_TIMEOUT_SECONDS) as client:
                r = await client.get(url, headers=headers, params=params)
                r.raise_for_status()
                return r.json()
        except httpx.HTTPStatusError as e:
            if attempt == RETRY_MAX_ATTEMPTS or (e.response is not None and e.response.status_code not in TRANSIENT_STATUS):
                raise
        except httpx.RequestError:
            if attempt == RETRY_MAX_ATTEMPTS:
                raise
        await asyncio.sleep(delay + random.uniform(0, delay * 0.25))
        delay *= 2

# -----------------------------------------------------------------------------
# Solana JSON-RPC helper
# -----------------------------------------------------------------------------
_SOL_RPC_CONCURRENCY = int(os.getenv("SOL_RPC_CONCURRENCY", "8"))
_sol_rpc_sem = asyncio.Semaphore(_SOL_RPC_CONCURRENCY)

async def sol_rpc(method: str, params: list) -> dict:
    if not SOLANA_RPC_URLS:
        raise HTTPException(status_code=500, detail="No SOLANA_RPC_URLS configured")
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    delay = RETRY_BASE_DELAY_MS / 1000.0

    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        rpc_url = SOLANA_RPC_URLS[(attempt - 1) % len(SOLANA_RPC_URLS)]
        try:
            async with _sol_rpc_sem:
                async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
                    r = await client.post(rpc_url, json=payload, headers={"Content-Type": "application/json"})
            if r.status_code in TRANSIENT_STATUS:
                await asyncio.sleep(0.75)
                continue
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "error" in data:
                code = data["error"].get("code")
                if code in JSONRPC_RATE_LIMIT_CODES:
                    await asyncio.sleep(0.75)
                    continue
            return data
        except httpx.RequestError:
            pass
        await asyncio.sleep(delay + random.uniform(0, delay * 0.25))
        delay = min(delay * 2, 8.0)

    raise HTTPException(status_code=502, detail=f"All Solana RPC endpoints failed for {method}")
# -----------------------------------------------------------------------------
# Jupiter token list cache
# -----------------------------------------------------------------------------
_JUP_CACHE: Optional[list] = None
_JUP_CACHE_TIME: float = 0
_JUP_CACHE_TTL: int = 300  # seconds

async def get_jup_tokens(timeout: float = 5.0) -> list:
    """Fetch Jupiter token list with caching and timeout."""
    global _JUP_CACHE, _JUP_CACHE_TIME
    now = time.time()
    if _JUP_CACHE and (now - _JUP_CACHE_TIME) < _JUP_CACHE_TTL:
        return _JUP_CACHE
    try:
        _JUP_CACHE = await http_get_json("https://token.jup.ag/all", timeout=timeout)
        _JUP_CACHE_TIME = now
    except Exception as e:
        print(f"[WARN] Jupiter fetch failed: {e} — using stale cache if available")
        return _JUP_CACHE or []
    return _JUP_CACHE or []

# -----------------------------------------------------------------------------
# Solana metadata enrichment (Pro → Public → Jupiter fallback)
# -----------------------------------------------------------------------------
async def sol_enrich_metadata(holdings: List[TokenHolding]) -> None:
    if not holdings:
        return

    sem = asyncio.Semaphore(8)  # limit concurrent HTTP calls
    jup_tokens = await get_jup_tokens(timeout=5.0)

    async def _enrich_single(h: TokenHolding):
        data = None

        # Try Pro endpoint first
        if SOLSCAN_API_TOKEN:
            try:
                async with sem:
                    data = await http_get_json(
                        f"https://pro-api.solscan.io/v2.0/token/meta?tokenAddress={h.mint}",
                        headers={"token": SOLSCAN_API_TOKEN}
                    )
                if isinstance(data, dict) and "error_message" in data and "Unauthorized" in data["error_message"]:
                    print(f"[WARN] Pro endpoint unauthorized for {h.mint}, falling back to public")
                    data = None
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    print(f"[WARN] Pro endpoint returned 401 for {h.mint}, falling back to public")
                    data = None
                else:
                    raise

        # Fallback to public endpoint
        if not data:
            try:
                async with sem:
                    data = await http_get_json(f"https://public-api.solscan.io/token/meta?tokenAddress={h.mint}")
                print(f"[DEBUG] Public Solscan response for {h.mint}: {data}")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    print(f"[WARN] Public endpoint returned 404 for {h.mint}, trying Jupiter fallback")
                    data = {}
                else:
                    raise

        # Ensure we only call .get() on dicts
        if isinstance(data, dict):
            v = data.get("data") or data or {}
        else:
            print(f"[WARN] Unexpected Solscan response type for {h.mint}: {type(data).__name__}")
            v = {}

        # Helper to detect placeholder values
        def is_placeholder(val: str, mint_addr: str) -> bool:
            return not val or str(val).strip().lower() == mint_addr.lower()

        # If still missing or placeholder, try Jupiter token list
        if jup_tokens and (
            is_placeholder(v.get("name"), h.mint) or
            is_placeholder(v.get("symbol"), h.mint) or
            not v.get("icon")
        ):
            match = next((t for t in jup_tokens if t.get("address") == h.mint), None)
            if match:
                v["name"] = match.get("name") or v.get("name")
                v["symbol"] = match.get("symbol") or v.get("symbol")
                v["icon"] = match.get("logoURI") or v.get("icon")

        # Apply enrichments
        h.name = v.get("name") or h.name
        h.symbol = v.get("symbol") or h.symbol
        h.logo_uri = v.get("icon") or h.logo_uri

    try:
        await asyncio.gather(*(_enrich_single(h) for h in holdings))
    except asyncio.CancelledError:
        print("[WARN] sol_enrich_metadata cancelled — returning partial results")
# -----------------------------------------------------------------------------
# Solana trace
# -----------------------------------------------------------------------------
async def sol_trace_wallet(wallet: str) -> List[TokenHolding]:
    data = await sol_rpc(
        "getTokenAccountsByOwner",
        [wallet, {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"}, {"encoding": "jsonParsed"}],
    )
    raw: List[TokenHolding] = []
    for item in data.get("result", {}).get("value", []):
        try:
            info = item["account"]["data"]["parsed"]["info"]
            mint = info["mint"]
            ta = info["tokenAmount"]
            amount = int(ta.get("amount", "0"))
            decimals = int(ta.get("decimals", 0))
            ui_amount = amount / (10 ** decimals) if decimals else float(amount)
            if amount > 0:
                raw.append(TokenHolding(mint=mint, amount=amount, decimals=decimals, ui_amount=ui_amount))
        except Exception:
            continue

    # Aggregate by mint
    by_mint: Dict[str, TokenHolding] = {}
    for h in raw:
        if h.mint not in by_mint:
            by_mint[h.mint] = h
        else:
            agg = by_mint[h.mint]
            agg.amount += h.amount
            agg.ui_amount = agg.amount / (10 ** agg.decimals) if agg.decimals else float(agg.amount)

    holdings = list(by_mint.values())

    # Enrich metadata for all holdings
    await sol_enrich_metadata(holdings)
    for h in holdings:
        _fill_metadata_defaults(h)

    holdings.sort(key=lambda x: x.ui_amount, reverse=True)
    return holdings

# -----------------------------------------------------------------------------
# Solana analyze (Pro → Public → Jupiter fallback + enrichment) — safe version + top holder breakdown
# -----------------------------------------------------------------------------
async def sol_analyze_mint(mint: str) -> AnalyzeResult:
    try:
        supply_data = await sol_rpc("getTokenSupply", [mint])
        supply_val = (supply_data.get("result") or {}).get("value") or {}
        supply_amount_raw = int(supply_val.get("amount") or 0)
        decimals = int(supply_val.get("decimals") or 0)
    except Exception as e:
        print(f"[ERROR] Failed to fetch supply for {mint}: {e}")
        supply_amount_raw, decimals = 0, 0

    holder_breakdown: List[Dict[str, Any]] = []
    try:
        largest_data = await sol_rpc("getTokenLargestAccounts", [mint, {"commitment": "finalized"}])
        largest = (largest_data.get("result") or {}).get("value") or []
        top_token_account = largest[0]["address"] if largest else None
        top_amount_raw = int(largest[0].get("amount") or 0) if largest else 0

        # Build breakdown for top N holders
        if isinstance(largest, list) and supply_amount_raw:
            for entry in largest[:10]:
                try:
                    addr = entry.get("address")
                    amt_raw = int(entry.get("amount") or 0)
                    pct = (amt_raw / float(supply_amount_raw)) * 100
                    holder_breakdown.append({
                        "address": addr,
                        "amount": amt_raw,
                        "pct": round(pct, 4)
                    })
                except Exception as e:
                    print(f"[WARN] Failed to parse holder entry: {e}")
    except Exception as e:
        print(f"[ERROR] Failed to fetch largest accounts for {mint}: {e}")
        top_token_account, top_amount_raw = None, 0

    top_owner = None
    if top_token_account:
        try:
            acct_data = await sol_rpc("getAccountInfo", [top_token_account, {"encoding": "jsonParsed"}])
            parsed = (((acct_data.get("result") or {}).get("value") or {}).get("data") or {}).get("parsed") or {}
            info = parsed.get("info") or {}
            top_owner = info.get("owner")
        except Exception as e:
            print(f"[WARN] Failed to fetch top owner for {mint}: {e}")

    # Risk scoring
    reasons: List[str] = []
    events: List[Dict[str, Any]] = []
    pct = (top_amount_raw / float(supply_amount_raw)) if supply_amount_raw else 0.0
    if pct >= 0.5:
        reasons.append("Top holder controls ≥ 50% of supply")
    elif pct >= 0.2:
        reasons.append("Top holder controls ≥ 20% of supply")
    elif pct >= 0.1:
        reasons.append("Top holder controls ≥ 10% of supply")
    penalty = int(min(100, round(pct * 100)))
    score = max(0, 100 - penalty)

    # Burn detection via Redis
    try:
        if redis_client:
            prev_supply = await get_previous_supply("solana", mint)
            await set_current_supply("solana", mint, supply_amount_raw)
            if prev_supply and supply_amount_raw < prev_supply:
                burn_pct = 100 * (prev_supply - supply_amount_raw) / prev_supply
                if burn_pct >= 10:
                    events.append({
                        "type": "burn",
                        "pct": round(burn_pct, 2),
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    })
                    reasons.append(f"Detected token burn: supply dropped {round(burn_pct, 2)}%")
    except Exception as e:
        print(f"[WARN] Burn detection failed for {mint}: {e}")

    # Build a temporary holding for enrichment
    holding = TokenHolding(
        mint=mint,
        amount=top_amount_raw,
        decimals=decimals,
        ui_amount=(top_amount_raw / (10 ** decimals)) if decimals else float(top_amount_raw),
    )
    try:
        await sol_enrich_metadata([holding])
    except Exception as e:
        print(f"[WARN] Metadata enrichment failed for {mint}: {e}")

    # Apply defaults if enrichment didn't fill them
    _fill_metadata_defaults(holding)
    if not holding.logo_uri:
        holding.logo_uri = PLACEHOLDER_LOGO

    metadata = Metadata(
        mint=mint,
        owner=top_owner,
        token_amount=top_amount_raw,
        decimals=decimals,
        supply=supply_amount_raw,
        name=holding.name or _sym_from_mint(mint),
        symbol=holding.symbol or _sym_from_mint(mint),
        logo_uri=holding.logo_uri,
    )
    risk = RiskScore(score=score if reasons else 60, reasons=reasons)

    return AnalyzeResult(
        address=mint,
        chain="solana",
        metadata=metadata,
        risk_score=risk,
        events=events or None,
        holders=holder_breakdown  # always a list, even if empty
    )
# -----------------------------------------------------------------------------
# Redis-backed supply snapshot helpers
# -----------------------------------------------------------------------------
async def get_previous_supply(chain: str, mint: str) -> Optional[int]:
    """Retrieve the last recorded supply for a given chain/mint from Redis."""
    if not redis_client:
        return None
    key = f"supply:{chain}:{mint}"
    val = await redis_client.get(key)
    try:
        return int(val) if val else None
    except Exception:
        return None

async def set_current_supply(chain: str, mint: str, supply: int) -> None:
    """Store the current supply for a given chain/mint in Redis with a 24h TTL."""
    if not redis_client:
        return
    key = f"supply:{chain}:{mint}"
    await redis_client.set(key, str(supply), ex=86400)
