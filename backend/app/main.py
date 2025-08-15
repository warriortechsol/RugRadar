import os
import random
import re
import asyncio
from typing import Any, Dict, List, Optional, Union, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Load .env BEFORE reading any environment variables
# -----------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
PORT = int(os.getenv("PORT", "8000"))
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "20"))
RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
RETRY_BASE_DELAY_MS = int(os.getenv("RETRY_BASE_DELAY_MS", "200"))

ALCHEMY_ETH_HTTP_URL = os.getenv("ALCHEMY_ETH_HTTP_URL", "").strip()
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY", "").strip()

SOLANA_RPC_URLS = [
    url.strip()
    for url in os.getenv("SOLANA_RPC_URLS", "https://api.mainnet-beta.solana.com").split(",")
    if url.strip()
]
SOLSCAN_API_TOKEN = os.getenv("SOLSCAN_API_TOKEN", "").strip()

CORS_ALLOW_ORIGINS = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")]

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

class TokenHolding(BaseModel):
    mint: str
    amount: int
    decimals: int
    ui_amount: float
    name: Optional[str] = None
    symbol: Optional[str] = None
    logo_uri: Optional[str] = None

class TraceResult(BaseModel):
    wallet: str
    chain: str
    tokens: List[TokenHolding]

class ChainsResponse(BaseModel):
    available: List[str]

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(title="RugRadar API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS if CORS_ALLOW_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def is_eth_address(addr: str) -> bool:
    return bool(re.fullmatch(r"0x[a-fA-F0-9]{40}", addr))

def _sym_from_mint(mint: str) -> str:
    return f"{mint[:4]}…{mint[-4:]}" if isinstance(mint, str) and len(mint) >= 8 else (mint or "UNK")

def _fill_metadata_defaults(h: TokenHolding) -> None:
    if not h.symbol or not str(h.symbol).strip():
        h.symbol = _sym_from_mint(h.mint)
    if not h.name or not str(h.name).strip():
        h.name = h.symbol

# -----------------------------------------------------------------------------
# HTTP client
# -----------------------------------------------------------------------------
TRANSIENT_STATUS = {500, 502, 503, 504}

async def http_post_json(
    url: Union[str, List[str]],
    json: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
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
            status = e.response.status_code if e.response is not None else None
            if attempt == RETRY_MAX_ATTEMPTS or status not in TRANSIENT_STATUS:
                raise
        except httpx.RequestError:
            if attempt == RETRY_MAX_ATTEMPTS:
                raise
        jitter = random.uniform(0, delay * 0.25)
        await asyncio.sleep(delay + jitter)
        delay *= 2

async def http_get_json(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    delay = RETRY_BASE_DELAY_MS / 1000.0
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
                r = await client.get(url, headers=headers, params=params)
                r.raise_for_status()
                return r.json()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            if attempt == RETRY_MAX_ATTEMPTS or status not in TRANSIENT_STATUS:
                raise
        except httpx.RequestError:
            if attempt == RETRY_MAX_ATTEMPTS:
                raise
        jitter = random.uniform(0, delay * 0.25)
        await asyncio.sleep(delay + jitter)
        delay *= 2

# -----------------------------------------------------------------------------
# Solana RPC + trace
# -----------------------------------------------------------------------------
async def sol_rpc(method: str, params: List[Any]) -> Dict[str, Any]:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    return await http_post_json(SOLANA_RPC_URLS, payload)

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

    by_mint: Dict[str, TokenHolding] = {}
    for h in raw:
        if h.mint not in by_mint:
            by_mint[h.mint] = h
        else:
            agg = by_mint[h.mint]
            agg.amount += h.amount
            agg.ui_amount = agg.amount / (10 ** agg.decimals) if agg.decimals else float(agg.amount)

    holdings = list(by_mint.values())
    await sol_enrich_metadata(holdings)
    for h in holdings:
        _fill_metadata_defaults(h)
    holdings.sort(key=lambda x: x.ui_amount, reverse=True)
    return holdings
# -----------------------------------------------------------------------------
# Solana enrich + analyze
# -----------------------------------------------------------------------------
async def sol_analyze_mint(mint: str) -> AnalyzeResult:
    supply_data = await sol_rpc("getTokenSupply", [mint])
    supply_val = (supply_data.get("result") or {}).get("value") or {}
    supply_amount_raw = int(supply_val.get("amount", "0"))
    decimals = int(supply_val.get("decimals", 0))

    largest_data = await sol_rpc("getTokenLargestAccounts", [mint, {"commitment": "finalized"}])
    largest = (largest_data.get("result") or {}).get("value") or []
    top_token_account = largest[0]["address"] if largest else None
    top_amount_raw = int(largest[0].get("amount", "0")) if largest else 0

    top_owner = None
    if top_token_account:
        acct_data = await sol_rpc("getAccountInfo", [top_token_account, {"encoding": "jsonParsed"}])
        parsed = (((acct_data.get("result") or {}).get("value") or {}).get("data") or {}).get("parsed") or {}
        info = parsed.get("info") or {}
        top_owner = info.get("owner")

    # Risk analysis
    reasons: List[str] = []
    pct = top_amount_raw / float(supply_amount_raw) if supply_amount_raw else 0.0
    if pct >= 0.5:
        reasons.append("Top holder controls ≥ 50% of supply")
    elif pct >= 0.2:
        reasons.append("Top holder controls ≥ 20% of supply")
    elif pct >= 0.1:
        reasons.append("Top holder controls ≥ 10% of supply")
    penalty = int(min(100, round(pct * 100)))
    score = max(0, 100 - penalty)

    # Hardened single‑mint metadata fetch
    name = symbol = logo = None
    if SOLSCAN_API_TOKEN:
        try:
            meta = await http_get_json(
                f"https://pro-api.solscan.io/v2.0/token/meta?tokenAddress={mint}",
                headers={"token": SOLSCAN_API_TOKEN},
            )
            mv = meta.get("data") or {}
            name = (mv.get("name") or "").strip() or name
            symbol = (mv.get("symbol") or "").strip() or symbol
            logo = (mv.get("icon") or "").strip() or logo
        except Exception:
            pass

    if not symbol:
        symbol = _sym_from_mint(mint)
    if not name:
        name = symbol
    if not logo:
        logo = "https://yourcdn.example.com/placeholder.png"

    metadata = Metadata(
        mint=mint,
        owner=top_owner,
        token_amount=top_amount_raw,
        decimals=decimals,
        supply=supply_amount_raw,
        name=name,
        symbol=symbol,
        logo_uri=logo,
    )
    risk = RiskScore(score=score, reasons=reasons)
    return AnalyzeResult(address=mint, chain="solana", metadata=metadata, risk_score=risk)

# -----------------------------------------------------------------------------
# Ethereum helpers
# -----------------------------------------------------------------------------
def _hex_to_int(x: str) -> int:
    return int(x, 16) if isinstance(x, str) and x.startswith("0x") else int(x or 0)

async def eth_get_token_balances(address: str) -> List[Dict[str, Any]]:
    if not ALCHEMY_ETH_HTTP_URL:
        raise HTTPException(status_code=500, detail="Alchemy ETH URL not configured")
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "alchemy_getTokenBalances",
        "params": [address, "erc20"],
    }
    data = await http_post_json(ALCHEMY_ETH_HTTP_URL, payload)
    return data.get("result", {}).get("tokenBalances", [])

async def eth_get_token_metadata(contract: str) -> Dict[str, Any]:
    if not ALCHEMY_ETH_HTTP_URL:
        raise HTTPException(status_code=500, detail="Alchemy ETH URL not configured")
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "alchemy_getTokenMetadata",
        "params": [contract],
    }
    data = await http_post_json(ALCHEMY_ETH_HTTP_URL, payload)
    return data.get("result", {}) or {}

async def eth_call(to: str, data: str) -> str:
    if not ALCHEMY_ETH_HTTP_URL:
        raise HTTPException(status_code=500, detail="Alchemy ETH URL not configured")
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_call",
        "params": [{"to": to, "data": data}, "latest"],
    }
    data = await http_post_json(ALCHEMY_ETH_HTTP_URL, payload)
    return (data.get("result") or "0x0").lower()

async def eth_get_total_supply_and_owner(contract: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    # totalSupply()
    total_supply_hex = await eth_call(contract, "0x18160ddd")
    total_supply = _hex_to_int(total_supply_hex) if total_supply_hex.startswith("0x") else None

    # owner() (Ownable) — best-effort
    try:
        owner_hex = await eth_call(contract, "0x8da5cb5b")
        if owner_hex and owner_hex.startswith("0x") and len(owner_hex) >= 66:
            owner_addr = "0x" + owner_hex[-40:]
        else:
            owner_addr = None
    except Exception:
        owner_addr = None

    decimals = None  # merged later with metadata.decimals
    return total_supply, decimals, owner_addr

async def eth_get_top_holders(contract: str) -> List[Dict[str, Any]]:
    # Placeholder to avoid external paid APIs; risk model handles empty list.
    return []

# -----------------------------------------------------------------------------
# Ethereum trace + analyze
# -----------------------------------------------------------------------------
async def eth_trace_wallet(wallet: str) -> List[TokenHolding]:
    balances = await eth_get_token_balances(wallet)
    out: List[TokenHolding] = []
    for b in balances:
        try:
            contract = b.get("contractAddress")
            bal_hex = b.get("tokenBalance")
            amount_raw = _hex_to_int(bal_hex or "0x0")
            if amount_raw == 0 or not contract:
                continue
            meta = await eth_get_token_metadata(contract)
            decimals = int(meta.get("decimals") or 0)
            ui = amount_raw / (10 ** decimals) if decimals else float(amount_raw)
            th = TokenHolding(
                mint=contract,
                amount=amount_raw,
                decimals=decimals,
                ui_amount=ui,
                name=meta.get("name"),
                symbol=meta.get("symbol"),
                logo_uri=meta.get("logo"),
            )
            _fill_metadata_defaults(th)
            out.append(th)
        except Exception:
            continue
    out.sort(key=lambda x: x.ui_amount, reverse=True)
    return out

async def eth_analyze_token(contract: str) -> AnalyzeResult:
    meta = await eth_get_token_metadata(contract)
    decimals_meta = int(meta.get("decimals") or 0)
    name = meta.get("name")
    symbol = meta.get("symbol")
    logo = meta.get("logo")

    total_supply, decimals_rpc, owner_addr = await eth_get_total_supply_and_owner(contract)
    decimals = decimals_rpc if decimals_rpc is not None else decimals_meta

    reasons: List[str] = []
    score = 100
    top_holders = await eth_get_top_holders(contract)

    if total_supply and top_holders:
        try:
            total_supply_float = float(total_supply) / (10 ** (decimals or 0))
        except Exception:
            total_supply_float = None
        if total_supply_float:
            try:
                top_bal = float(top_holders[0].get("Quantity") or 0)
                pct = top_bal / total_supply_float
                if pct >= 0.5:
                    reasons.append("Top holder controls ≥ 50% of supply")
                elif pct >= 0.2:
                    reasons.append("Top holder controls ≥ 20% of supply")
                elif pct >= 0.1:
                    reasons.append("Top holder controls ≥ 10% of supply")
                penalty = int(min(100, round(pct * 100)))
                score = max(0, 100 - penalty)
            except Exception:
                pass

    if not symbol:
        symbol = _sym_from_mint(contract)
    if not name:
        name = symbol

    metadata = Metadata(
        mint=contract,
        owner=owner_addr,
        token_amount=None,
        decimals=decimals,
        supply=total_supply,
        name=name,
        symbol=symbol,
        logo_uri=logo,
    )
    risk = RiskScore(score=score if reasons else 60, reasons=reasons or ["Baseline score pending deeper holder analysis"])
    return AnalyzeResult(address=contract, chain="ethereum", metadata=metadata, risk_score=risk)
import os
import re
import random
import asyncio
from typing import Any, Dict, List, Optional, Union, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Load .env BEFORE reading any environment variables
# -----------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
PORT = int(os.getenv("PORT", "8000"))
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "20"))
RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
RETRY_BASE_DELAY_MS = int(os.getenv("RETRY_BASE_DELAY_MS", "200"))

ALCHEMY_ETH_HTTP_URL = os.getenv("ALCHEMY_ETH_HTTP_URL", "").strip()
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY", "").strip()

SOLANA_RPC_URLS = [
    url.strip()
    for url in os.getenv("SOLANA_RPC_URLS", "https://api.mainnet-beta.solana.com").split(",")
    if url.strip()
]
SOLSCAN_API_TOKEN = os.getenv("SOLSCAN_API_TOKEN", "").strip()

CORS_ALLOW_ORIGINS = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")]

PLACEHOLDER_LOGO = os.getenv("PLACEHOLDER_LOGO", "https://yourcdn.example.com/placeholder.png").strip()

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

class TokenHolding(BaseModel):
    mint: str
    amount: int
    decimals: int
    ui_amount: float
    name: Optional[str] = None
    symbol: Optional[str] = None
    logo_uri: Optional[str] = None

class TraceResult(BaseModel):
    wallet: str
    chain: str
    tokens: List[TokenHolding]

class ChainsResponse(BaseModel):
    available: List[str]

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(title="RugRadar API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS if CORS_ALLOW_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    # Base58 (no 0, O, I, l), typical 32-44 chars
    return bool(re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{32,44}", addr))

def is_eth_address(addr: str) -> bool:
    return bool(re.fullmatch(r"0x[a-fA-F0-9]{40}", addr))

def _sym_from_mint(mint: str) -> str:
    return f"{mint[:4]}…{mint[-4:]}" if isinstance(mint, str) and len(mint) >= 8 else (mint or "UNK")

def _fill_metadata_defaults(h: TokenHolding) -> None:
    if not h.symbol or not str(h.symbol).strip():
        h.symbol = _sym_from_mint(h.mint)
    if not h.name or not str(h.name).strip():
        h.name = h.symbol
    if not h.logo_uri or not str(h.logo_uri).strip():
        h.logo_uri = PLACEHOLDER_LOGO

# -----------------------------------------------------------------------------
# HTTP client with retry + jitter
# -----------------------------------------------------------------------------
TRANSIENT_STATUS = {500, 502, 503, 504}

async def http_post_json(
    url: Union[str, List[str]],
    json: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
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
            status = e.response.status_code if e.response is not None else None
            if attempt == RETRY_MAX_ATTEMPTS or status not in TRANSIENT_STATUS:
                raise
        except httpx.RequestError:
            if attempt == RETRY_MAX_ATTEMPTS:
                raise
        jitter = random.uniform(0, delay * 0.25)
        await asyncio.sleep(delay + jitter)
        delay *= 2

async def http_get_json(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    delay = RETRY_BASE_DELAY_MS / 1000.0
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
                r = await client.get(url, headers=headers, params=params)
                r.raise_for_status()
                return r.json()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            if attempt == RETRY_MAX_ATTEMPTS or status not in TRANSIENT_STATUS:
                raise
        except httpx.RequestError:
            if attempt == RETRY_MAX_ATTEMPTS:
                raise
        jitter = random.uniform(0, delay * 0.25)
        await asyncio.sleep(delay + jitter)
        delay *= 2

# -----------------------------------------------------------------------------
# Solana RPC + trace
# -----------------------------------------------------------------------------
async def sol_rpc(method: str, params: List[Any]) -> Dict[str, Any]:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    return await http_post_json(SOLANA_RPC_URLS, payload)

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
    await sol_enrich_metadata(holdings)
    for h in holdings:
        _fill_metadata_defaults(h)
    holdings.sort(key=lambda x: x.ui_amount, reverse=True)
    return holdings

# -----------------------------------------------------------------------------
# Solana enrich + analyze
# -----------------------------------------------------------------------------
async def sol_enrich_metadata(holdings: List[TokenHolding]) -> None:
    if not holdings:
        return
    # Prefer Pro endpoint when token present; otherwise use public
    base_url = "https://pro-api.solscan.io/v2.0" if SOLSCAN_API_TOKEN else "https://public-api.solscan.io"
    headers = {"token": SOLSCAN_API_TOKEN} if SOLSCAN_API_TOKEN else None

    for h in holdings:
        try:
            url = f"{base_url}/token/meta?tokenAddress={h.mint}"
            data = await http_get_json(url, headers=headers)
            obj = (data.get("data") if isinstance(data, dict) else None) or data or {}
            h.name = (str(obj.get("name") or "").strip()) or h.name
            h.symbol = (str(obj.get("symbol") or "").strip()) or h.symbol
            h.logo_uri = (str(obj.get("icon") or obj.get("logoURI") or obj.get("logo") or "").strip()) or h.logo_uri
        except Exception:
            continue
        if not h.logo_uri:
            h.logo_uri = PLACEHOLDER_LOGO

async def sol_analyze_mint(mint: str) -> AnalyzeResult:
    supply_data = await sol_rpc("getTokenSupply", [mint])
    supply_val = (supply_data.get("result") or {}).get("value") or {}
    supply_amount_raw = int(supply_val.get("amount", "0"))
    decimals = int(supply_val.get("decimals", 0))

    largest_data = await sol_rpc("getTokenLargestAccounts", [mint, {"commitment": "finalized"}])
    largest = (largest_data.get("result") or {}).get("value") or []
    top_token_account = largest[0]["address"] if largest else None
    top_amount_raw = int(largest[0].get("amount", "0")) if largest else 0

    top_owner = None
    if top_token_account:
        acct_data = await sol_rpc("getAccountInfo", [top_token_account, {"encoding": "jsonParsed"}])
        parsed = (((acct_data.get("result") or {}).get("value") or {}).get("data") or {}).get("parsed") or {}
        info = parsed.get("info") or {}
        top_owner = info.get("owner")

    # Risk analysis (concentration-based)
    reasons: List[str] = []
    pct = top_amount_raw / float(supply_amount_raw) if supply_amount_raw else 0.0
    if pct >= 0.5:
        reasons.append("Top holder controls ≥ 50% of supply")
    elif pct >= 0.2:
        reasons.append("Top holder controls ≥ 20% of supply")
    elif pct >= 0.1:
        reasons.append("Top holder controls ≥ 10% of supply")
    penalty = int(min(100, round(pct * 100)))
    score = max(0, 100 - penalty)

    # Hardened single‑mint metadata fetch with Pro → Public fallback
    name = symbol = logo = None
    try:
        if SOLSCAN_API_TOKEN:
            url = f"https://pro-api.solscan.io/v2.0/token/meta?tokenAddress={mint}"
            meta = await http_get_json(url, headers={"token": SOLSCAN_API_TOKEN})
        else:
            url = f"https://public-api.solscan.io/token/meta?tokenAddress={mint}"
            meta = await http_get_json(url)
        mv = (meta.get("data") if isinstance(meta, dict) else None) or meta or {}
        name = (str(mv.get("name") or "").strip()) or name
        symbol = (str(mv.get("symbol") or "").strip()) or symbol
        logo = (str(mv.get("icon") or mv.get("logoURI") or mv.get("logo") or "").strip()) or logo
    except Exception:
        pass

    if not symbol:
        symbol = _sym_from_mint(mint)
    if not name:
        name = symbol
    if not logo:
        logo = PLACEHOLDER_LOGO

    metadata = Metadata(
        mint=mint,
        owner=top_owner,
        token_amount=top_amount_raw,
        decimals=decimals,
        supply=supply_amount_raw,
        name=name,
        symbol=symbol,
        logo_uri=logo,
    )
    risk = RiskScore(score=score, reasons=reasons)
    return AnalyzeResult(address=mint, chain="solana", metadata=metadata, risk_score=risk)

# -----------------------------------------------------------------------------
# Ethereum helpers
# -----------------------------------------------------------------------------
def _hex_to_int(x: str) -> int:
    return int(x, 16) if isinstance(x, str) and x.startswith("0x") else int(x or 0)

async def eth_get_token_balances(address: str) -> List[Dict[str, Any]]:
    if not ALCHEMY_ETH_HTTP_URL:
        raise HTTPException(status_code=500, detail="Alchemy ETH URL not configured")
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "alchemy_getTokenBalances",
        "params": [address, "erc20"],
    }
    data = await http_post_json(ALCHEMY_ETH_HTTP_URL, payload)
    return data.get("result", {}).get("tokenBalances", [])

async def eth_get_token_metadata(contract: str) -> Dict[str, Any]:
    if not ALCHEMY_ETH_HTTP_URL:
        raise HTTPException(status_code=500, detail="Alchemy ETH URL not configured")
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "alchemy_getTokenMetadata",
        "params": [contract],
    }
    data = await http_post_json(ALCHEMY_ETH_HTTP_URL, payload)
    return data.get("result", {}) or {}

async def eth_call(to: str, data: str) -> str:
    if not ALCHEMY_ETH_HTTP_URL:
        raise HTTPException(status_code=500, detail="Alchemy ETH URL not configured")
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_call",
        "params": [{"to": to, "data": data}, "latest"],
    }
    data = await http_post_json(ALCHEMY_ETH_HTTP_URL, payload)
    return (data.get("result") or "0x0").lower()

async def eth_get_total_supply_and_owner(contract: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    # totalSupply()
    total_supply_hex = await eth_call(contract, "0x18160ddd")
    total_supply = _hex_to_int(total_supply_hex) if total_supply_hex.startswith("0x") else None

    # owner() (Ownable) — best-effort
    try:
        owner_hex = await eth_call(contract, "0x8da5cb5b")
        if owner_hex and owner_hex.startswith("0x") and len(owner_hex) >= 66:
            owner_addr = "0x" + owner_hex[-40:]
        else:
            owner_addr = None
    except Exception:
        owner_addr = None

    decimals = None  # merged later with metadata.decimals
    return total_supply, decimals, owner_addr

async def eth_get_top_holders(contract: str) -> List[Dict[str, Any]]:
    # Placeholder to avoid external paid APIs; risk model handles empty list.
    return []

# -----------------------------------------------------------------------------
# Ethereum trace + analyze
# -----------------------------------------------------------------------------
async def eth_trace_wallet(wallet: str) -> List[TokenHolding]:
    balances = await eth_get_token_balances(wallet)
    out: List[TokenHolding] = []
    for b in balances:
        try:
            contract = b.get("contractAddress")
            bal_hex = b.get("tokenBalance")
            amount_raw = _hex_to_int(bal_hex or "0x0")
            if amount_raw == 0 or not contract:
                continue
            meta = await eth_get_token_metadata(contract)
            decimals = int(meta.get("decimals") or 0)
            ui = amount_raw / (10 ** decimals) if decimals else float(amount_raw)
            th = TokenHolding(
                mint=contract,
                amount=amount_raw,
                decimals=decimals,
                ui_amount=ui,
                name=meta.get("name"),
                symbol=meta.get("symbol"),
                logo_uri=meta.get("logo"),
            )
            _fill_metadata_defaults(th)
            out.append(th)
        except Exception:
            continue
    out.sort(key=lambda x: x.ui_amount, reverse=True)
    return out

async def eth_analyze_token(contract: str) -> AnalyzeResult:
    meta = await eth_get_token_metadata(contract)
    decimals_meta = int(meta.get("decimals") or 0)
    name = meta.get("name")
    symbol = meta.get("symbol")
    logo = meta.get("logo")

    total_supply, decimals_rpc, owner_addr = await eth_get_total_supply_and_owner(contract)
    decimals = decimals_rpc if decimals_rpc is not None else decimals_meta

    reasons: List[str] = []
    score = 100
    top_holders = await eth_get_top_holders(contract)

    if total_supply and top_holders:
        try:
            total_supply_float = float(total_supply) / (10 ** (decimals or 0))
        except Exception:
            total_supply_float = None
        if total_supply_float:
            try:
                top_bal = float(top_holders[0].get("Quantity") or 0)
                pct = top_bal / total_supply_float
                if pct >= 0.5:
                    reasons.append("Top holder controls ≥ 50% of supply")
                elif pct >= 0.2:
                    reasons.append("Top holder controls ≥ 20% of supply")
                elif pct >= 0.1:
                    reasons.append("Top holder controls ≥ 10% of supply")
                penalty = int(min(100, round(pct * 100)))
                score = max(0, 100 - penalty)
            except Exception:
                pass

    if not symbol:
        symbol = _sym_from_mint(contract)
    if not name:
        name = symbol
    if not logo:
        logo = PLACEHOLDER_LOGO

    metadata = Metadata(
        mint=contract,
        owner=owner_addr,
        token_amount=None,
        decimals=decimals,
        supply=total_supply,
        name=name,
        symbol=symbol,
        logo_uri=logo,
    )
    risk = RiskScore(score=score if reasons else 60, reasons=reasons or ["Baseline score pending deeper holder analysis"])
    return AnalyzeResult(address=contract, chain="ethereum", metadata=metadata, risk_score=risk)

# -----------------------------------------------------------------------------
# Normalization helpers
# -----------------------------------------------------------------------------
def _to_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    return jsonable_encoder(obj)

def normalize_analyze_result(res: Any, address: str, chain: str) -> Dict[str, Any]:
    d = _to_dict(res) or {}
    md = d.get("metadata") or {}
    rs = d.get("risk_score") or {}

    def _int(v, default=0):
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    # Apply durable fallbacks to ensure buyer-friendly output
    name_fallback = md.get("name") or md.get("symbol") or _sym_from_mint(md.get("mint") or address)
    symbol_fallback = md.get("symbol") or _sym_from_mint(md.get("mint") or address)
    logo_fallback = md.get("logo_uri") or PLACEHOLDER_LOGO

    return {
        "address": d.get("address") or address,
        "chain": d.get("chain") or chain,
        "metadata": {
            "mint": md.get("mint") or md.get("contract") or address,
            "owner": md.get("owner") or "",
            "token_amount": _int(md.get("token_amount")),
            "decimals": _int(md.get("decimals")),
            "supply": _int(md.get("supply")),
            "name": name_fallback,
            "symbol": symbol_fallback,
            "logo_uri": logo_fallback,
        },
        "risk_score": {
            "score": _int(rs.get("score")),
            "reasons": rs.get("reasons") or [],
        },
    }

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def ping():
    return {"message": "RugRadar backend online."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/chains", response_model=ChainsResponse)
async def get_chains() -> ChainsResponse:
    available = ["solana"]
    if ALCHEMY_ETH_HTTP_URL:
        available.append("ethereum")
    return ChainsResponse(available=available)

@app.get("/trace", response_model=TraceResult)
async def trace(
    wallet: Optional[str] = Query(default=None, description="Wallet address"),
    address: Optional[str] = Query(default=None, description="Alias for wallet"),
    chain: str = Query(..., description="Chain name: solana or ethereum")
) -> TraceResult:
    target = sanitize_address(wallet or address or "")
    chain_s = (chain or "").strip().lower()
    if not target:
        raise HTTPException(status_code=422, detail="Query param 'wallet' or 'address' is required")
    try:
        if chain_s == "solana":
            if not is_solana_address(target):
                raise HTTPException(status_code=422, detail="Invalid Solana wallet address")
            tokens = await sol_trace_wallet(target)
            return TraceResult(wallet=target, chain=chain_s, tokens=tokens)
        elif chain_s == "ethereum":
            if not is_eth_address(target):
                raise HTTPException(status_code=422, detail="Invalid Ethereum address")
            tokens = await eth_trace_wallet(target)
            return TraceResult(wallet=target, chain=chain_s, tokens=tokens)
        else:
            raise HTTPException(status_code=400, detail="Unsupported chain. Try chain=solana or chain=ethereum")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trace failure: {str(e)}")

@app.get("/analyze", response_model=AnalyzeResult)
async def analyze(
    address: Optional[str] = Query(None, description="Wallet or token/contract address"),
    wallet: Optional[str] = Query(None, description="Alias for address"),
    mint: Optional[str] = Query(None, description="Alias for token/mint"),
    chain: str = Query(..., description="Chain name: solana or ethereum"),
) -> AnalyzeResult:
    addr_s = sanitize_address(address or wallet or mint or "")
    chain_s = (chain or "").strip().lower()
    
    if not addr_s:
        raise HTTPException(status_code=422, detail="Query param 'wallet' or 'address' is required")
    
    try:
        if chain_s == "solana":
            if not is_solana_address(addr_s):
                raise HTTPException(status_code=422, detail="Invalid Solana mint/address")
            raw = await sol_analyze_mint(addr_s)
            return normalize_analyze_result(raw, addr_s, chain_s)  # type: ignore[return-value]

        elif chain_s == "ethereum":
            if not is_eth_address(addr_s):
                raise HTTPException(status_code=422, detail="Invalid Ethereum contract/address")
            raw = await eth_analyze_token(addr_s)
            return normalize_analyze_result(raw, addr_s, chain_s)  # type: ignore[return-value]

        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported chain. Try chain=solana or chain=ethereum"
            )

    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analyze failure: {str(e)}")


# -----------------------------------------------------------------------------
# Optional: run with uvicorn if executed directly
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=PORT, reload=False)
