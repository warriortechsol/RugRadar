import os
import random
import re
import asyncio
from typing import Any, Dict, List, Optional, Union

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
    allow_origins=[
        "https://rugradar-frontend.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}
    
# -----------------------------------------------------------------------------
# Helpers: validation & sanitization
# -----------------------------------------------------------------------------

def sanitize_address(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = s.split("#", 1)[0].split("?", 1)[0]
    # keep your current heuristic for pump.fun suffixes
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
    # Ensure symbol and name are never empty so the UI has clear labels
    if not h.symbol or not str(h.symbol).strip():
        h.symbol = _sym_from_mint(h.mint)
    if not h.name or not str(h.name).strip():
        h.name = h.symbol

# -----------------------------------------------------------------------------
# HTTP client with retry/backoff (async-safe with jitter)
# -----------------------------------------------------------------------------

TRANSIENT_STATUS = {500, 502, 503, 504}

async def http_post_json(
    url: Union[str, List[str]],
    json: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
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
    params: Optional[Dict[str, Any]] = None,
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
# Solana RPC/metadata
# -----------------------------------------------------------------------------

def pick_solana_rpc() -> str:
    return random.choice(SOLANA_RPC_URLS)

async def sol_rpc(method: str, params: List[Any]) -> Dict[str, Any]:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    # rotate across SOLANA_RPC_URLS per attempt inside http_post_json
    return await http_post_json(SOLANA_RPC_URLS, payload)

async def sol_trace_wallet(wallet: str) -> List[TokenHolding]:
    data = await sol_rpc(
        "getTokenAccountsByOwner",
        [
            wallet,
            {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
            {"encoding": "jsonParsed"},
        ],
    )
    raw_holdings: List[TokenHolding] = []
    for item in data.get("result", {}).get("value", []):
        try:
            info = item["account"]["data"]["parsed"]["info"]
            mint = info["mint"]
            ta = info["tokenAmount"]
            amount = int(ta.get("amount", "0"))
            decimals = int(ta.get("decimals", 0))
            ui_amount = float(ta.get("uiAmount", 0.0))
            if amount > 0:
                raw_holdings.append(TokenHolding(mint=mint, amount=amount, decimals=decimals, ui_amount=ui_amount))
        except Exception:
            continue

    # Aggregate by mint so tokens don't "run together" across multiple token accounts
    by_mint: Dict[str, TokenHolding] = {}
    for h in raw_holdings:
        if h.mint not in by_mint:
            by_mint[h.mint] = TokenHolding(
                mint=h.mint,
                amount=h.amount,
                decimals=h.decimals,
                ui_amount=h.ui_amount if h.ui_amount > 0 else (h.amount / (10 ** h.decimals) if h.decimals else float(h.amount)),
            )
        else:
            agg = by_mint[h.mint]
            agg.amount += h.amount
            # Decimals should be consistent per mint; keep existing
            agg.ui_amount = agg.amount / (10 ** agg.decimals) if agg.decimals else float(agg.amount)

    holdings: List[TokenHolding] = list(by_mint.values())

    # Enrich with Solscan metadata where available
    await sol_enrich_metadata(holdings)

    # Ensure defaults for name/symbol to avoid blank labels in UI
    for h in holdings:
        _fill_metadata_defaults(h)

    # Sort descending by ui_amount for consistent display
    holdings.sort(key=lambda x: x.ui_amount, reverse=True)
    return holdings

async def sol_enrich_metadata(holdings: List[TokenHolding]) -> None:
    if not holdings or not SOLSCAN_API_TOKEN:
        return
    headers = {"token": SOLSCAN_API_TOKEN}

    sem = asyncio.Semaphore(8)

    async def fetch_one(h: TokenHolding):
        url = f"https://pro-api.solscan.io/v2.0/token/meta?tokenAddress={h.mint}"
        try:
            async with sem:
                data = await http_get_json(url, headers=headers)
            v = data.get("data") or {}
            # Only set if present; defaults handled separately
            h.name = v.get("name") or h.name
            h.symbol = v.get("symbol") or h.symbol
            h.logo_uri = v.get("icon") or h.logo_uri
        except Exception:
            # Leave defaults to be filled later
            pass

    await asyncio.gather(*(fetch_one(h) for h in holdings))

async def sol_analyze_mint(mint: str) -> AnalyzeResult:
    # Supply
    supply_data = await sol_rpc("getTokenSupply", [mint])
    supply_value = (supply_data.get("result") or {}).get("value") or {}
    supply_amount_raw = int(supply_value.get("amount", "0"))
    decimals = int(supply_value.get("decimals", 0))

    # Largest accounts
    largest_data = await sol_rpc("getTokenLargestAccounts", [mint, {"commitment": "finalized"}])
    largest = (largest_data.get("result") or {}).get("value") or []
    top_token_account = largest[0]["address"] if largest else None
    top_amount_raw = int(largest[0].get("amount", "0")) if largest else 0

    # Resolve owner of top token account
    top_owner = None
    if top_token_account:
        acct_data = await sol_rpc("getAccountInfo", [top_token_account, {"encoding": "jsonParsed"}])
        parsed = (((acct_data.get("result") or {}).get("value") or {}).get("data") or {}).get("parsed") or {}
        info = parsed.get("info") or {}
        top_owner = info.get("owner")

    # Concentration heuristic
    reasons: List[str] = []
    pct = 0.0
    if supply_amount_raw > 0:
        pct = top_amount_raw / float(supply_amount_raw)
        if pct >= 0.5:
            reasons.append("Top holder controls ≥ 50% of supply")
        elif pct >= 0.2:
            reasons.append("Top holder controls ≥ 20% of supply")
        elif pct >= 0.1:
            reasons.append("Top holder controls ≥ 10% of supply")

    penalty = int(min(100, round(pct * 100)))
    score = max(0, 100 - penalty)

    # Basic metadata enrichment
    name = symbol = logo = None
    if SOLSCAN_API_TOKEN:
        try:
            meta = await http_get_json(
                f"https://pro-api.solscan.io/v2.0/token/meta?tokenAddress={mint}",
                headers={"token": SOLSCAN_API_TOKEN},
            )
            mv = meta.get("data") or {}
            name, symbol, logo = mv.get("name"), mv.get("symbol"), mv.get("icon")
        except Exception:
            pass

    # Fallbacks for metadata to avoid blank headings
    if not symbol:
        symbol = _sym_from_mint(mint)
    if not name:
        name = symbol

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
# Ethereum via Alchemy
# -----------------------------------------------------------------------------

def _hex_to_int(x: str) -> int:
    if isinstance(x, str) and x.startswith("0x"):
        return int(x, 16)
    return int(x)

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
    result = data.get("result") or {}
    return result.get("tokenBalances") or []

async def eth_get_token_metadata(contract: str) -> Dict[str, Any]:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "alchemy_getTokenMetadata",
        "params": [contract],
    }
    data = await http_post_json(ALCHEMY_ETH_HTTP_URL, payload)
    return data.get("result") or {}

async def eth_trace_wallet(wallet: str) -> List[TokenHolding]:
    balances = await eth_get_token_balances(wallet)
    out: List[TokenHolding] = []
    for b in balances:
        try:
            contract = b.get("contractAddress")
            bal_hex = b.get("tokenBalance")
            amount_raw = _hex_to_int(bal_hex or "0x0")
            if amount_raw == 0:
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
                logo_uri=meta.get("logo"),  # may be None
            )
            _fill_metadata_defaults(th)
            out.append(th)
        except Exception:
            continue
    # Sort descending by ui_amount for consistent display
    out.sort(key=lambda x: x.ui_amount, reverse=True)
    return out

# Placeholder analysis for ETH (extend in next iteration)
async def eth_analyze_token(contract: str) -> AnalyzeResult:
    # Minimal safe default: return supply/decimals via metadata, basic score
    meta = await eth_get_token_metadata(contract)
    decimals = int(meta.get("decimals") or 0)
    name = meta.get("name")
    symbol = meta.get("symbol")
    logo = meta.get("logo")

    # Fallbacks to keep headings clean
    if not symbol:
        symbol = _sym_from_mint(contract)
    if not name:
        name = symbol

    metadata = Metadata(mint=contract, decimals=decimals, name=name, symbol=symbol, logo_uri=logo)
    risk = RiskScore(score=60, reasons=["Baseline score pending deeper holder analysis"])
    return AnalyzeResult(address=contract, chain="ethereum", metadata=metadata, risk_score=risk)

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/chains", response_model=ChainsResponse)
async def get_chains() -> ChainsResponse:
    available = ["solana"]
    if ALCHEMY_ETH_HTTP_URL:
        available.append("ethereum")
    return ChainsResponse(available=available)

@app.get("/trace", response_model=TraceResult)
async def trace(wallet: str = Query(...), chain: str = Query(...)) -> TraceResult:
    wallet_s = sanitize_address(wallet)
    chain_s = (chain or "").strip().lower()

    try:
        if chain_s == "solana":
            if not is_solana_address(wallet_s):
                raise HTTPException(status_code=422, detail="Invalid Solana wallet address")
            tokens = await sol_trace_wallet(wallet_s)
            return TraceResult(wallet=wallet_s, chain=chain_s, tokens=tokens)

        elif chain_s == "ethereum":
            if not is_eth_address(wallet_s):
                raise HTTPException(status_code=422, detail="Invalid Ethereum address")
            tokens = await eth_trace_wallet(wallet_s)
            return TraceResult(wallet=wallet_s, chain=chain_s, tokens=tokens)

        else:
            raise HTTPException(status_code=400, detail="Unsupported chain. Try chain=solana or chain=ethereum")

    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trace failure: {str(e)}")

@app.get("/analyze", response_model=AnalyzeResult)
async def analyze(address: str = Query(...), chain: str = Query(...)) -> AnalyzeResult:
    addr_s = sanitize_address(address)
    chain_s = (chain or "").strip().lower()

    try:
        if chain_s == "solana":
            if not is_solana_address(addr_s):
                raise HTTPException(status_code=422, detail="Invalid Solana mint address")
            return await sol_analyze_mint(addr_s)

        if chain_s == "ethereum":
            if not is_eth_address(addr_s):
                # For Ethereum analyze, we accept contract addresses (same format as wallet)
                raise HTTPException(status_code=422, detail="Invalid Ethereum contract address")
            return await eth_analyze_token(addr_s)

        raise HTTPException(status_code=400, detail="Unsupported chain. Try chain=solana or chain=ethereum")

    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analyze failure: {str(e)}")
@app.get("/")
def ping():
    return {"message": "RugRadar backend online."}


