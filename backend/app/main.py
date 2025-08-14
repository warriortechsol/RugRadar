# backend/app/main.py

import os
import re
import random
import asyncio
from typing import Any, Dict, List, Optional, Union

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Load environment
# -----------------------------------------------------------------------------
load_dotenv()

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
PORT = int(os.getenv("PORT", "8000"))
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "20"))
RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
RETRY_BASE_DELAY_MS = int(os.getenv("RETRY_BASE_DELAY_MS", "200"))

ALCHEMY_ETH_URL = os.getenv("ALCHEMY_ETH_HTTP_URL", "").strip()
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
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def ping():
    return {"message": "RugRadar backend online."}

@app.get("/health")
def health():
    return {"status": "ok"}

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
            status = e.response.status_code if e.response is not None else None
            if attempt == RETRY_MAX_ATTEMPTS or status not in TRANSIENT_STATUS:
                raise
        except httpx.RequestError:
            if attempt == RETRY_MAX_ATTEMPTS:
                raise
        jitter = random.uniform(0, delay * 0.25)
        await asyncio.sleep(delay + jitter)
        delay *= 2

async def http_get_json(url: str, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
# Solana
# -----------------------------------------------------------------------------
def pick_solana_rpc() -> str:
    return random.choice(SOLANA_RPC_URLS)

async def sol_rpc(method: str, params: List[Any]) -> Dict[str, Any]:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    return await http_post_json(SOLANA_RPC_URLS, payload)

async def sol_trace_wallet(wallet: str) -> List[TokenHolding]:
    data = await sol_rpc("getTokenAccountsByOwner", [
        wallet,
        {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
        {"encoding": "jsonParsed"},
    ])
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

    by_mint: Dict[str, TokenHolding] = {}
    for h in raw_holdings:
        if h.mint not in by_mint:
            by_mint[h.mint] = h
        else:
            agg = by_mint[h.mint]
            agg.amount += h.amount
            agg.ui_amount = agg.amount / (10 ** agg.decimals) if agg.decimals else float(agg.amount)

    holdings = list(by_mint.values())
    for h in holdings:
        _fill_metadata_defaults(h)
    holdings.sort(key=lambda x: x.ui_amount, reverse=True)
    return holdings

# -----------------------------------------------------------------------------
# Ethereum
# -----------------------------------------------------------------------------
def _hex_to_int(x: str) -> int:
    return int(x, 16) if isinstance(x, str) and x.startswith("0x") else int(x)

async def eth_get_token_balances(address: str) -> List[Dict[str, Any]]:
    if not ALCHEMY_ETH_URL:
        raise HTTPException(status_code=500, detail="Alchemy ETH URL not configured")
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "alchemy_getTokenBalances",
        "params": [address, "erc20"],
    }
    data = await http_post_json(ALCHEMY_ETH_URL, payload)
    return data.get("result", {}).get("tokenBalances", [])


async def eth_get_token_metadata(contract: str) -> Dict[str, Any]:
    if not ALCHEMY_ETH_URL:
        raise HTTPException(status_code=500, detail="Alchemy ETH URL not configured")
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "alchemy_getTokenMetadata",
        "params": [contract],
    }
    data = await http_post_json(ALCHEMY_ETH_URL, payload)
    return data.get("result", {})

async def eth_trace_wallet(wallet: str) -> List[TokenHolding]:
    balances = await eth_get_token_balances(wallet)
    holdings: List[TokenHolding] = []
    for b in balances:
        try:
            raw_balance = _hex_to_int(b.get("tokenBalance", "0x0"))
            contract = b.get("contractAddress", "")
            if raw_balance == 0 or not contract:
                continue
            meta = await eth_get_token_metadata(contract)
            decimals = int(meta.get("decimals", 0))
            ui_amount = raw_balance / (10 ** decimals) if decimals else float(raw_balance)
            h = TokenHolding(
                mint=contract,
                amount=raw_balance,
                decimals=decimals,
                ui_amount=ui_amount,
                name=meta.get("name"),
                symbol=meta.get("symbol"),
                logo_uri=meta.get("logo"),
            )
            _fill_metadata_defaults(h)
            holdings.append(h)
        except Exception:
            continue
    holdings.sort(key=lambda x: x.ui_amount, reverse=True)
    return holdings

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/chains", response_model=ChainsResponse)
def get_chains():
    return ChainsResponse(available=["solana", "ethereum"])

@app.get("/trace", response_model=TraceResult)
async def trace_wallet(
    wallet: str = Query(..., description="Wallet address"),
    chain: str = Query(..., description="Chain name: solana or ethereum"),
):
    wallet = sanitize_address(wallet)
    chain = chain.lower()
    if chain == "solana":
        if not is_solana_address(wallet):
            raise HTTPException(status_code=400, detail="Invalid Solana address")
        tokens = await sol_trace_wallet(wallet)
    elif chain == "ethereum":
        if not is_eth_address(wallet):
            raise HTTPException(status_code=400, detail="Invalid Ethereum address")
        tokens = await eth_trace_wallet(wallet)
    else:
        raise HTTPException(status_code=400, detail="Unsupported chain")
    return TraceResult(wallet=wallet, chain=chain, tokens=tokens)
    
@app.get("/analyze", response_model=AnalyzeResult)
async def analyze_wallet(
    wallet: str = Query(..., description="Wallet address"),
    chain: str = Query(..., description="Chain name: solana or ethereum"),
):
    wallet = sanitize_address(wallet)
    chain = chain.lower()

    if chain == "solana":
        if not is_solana_address(wallet):
            raise HTTPException(status_code=400, detail="Invalid Solana address")
        tokens = await sol_trace_wallet(wallet)
    elif chain == "ethereum":
        if not is_eth_address(wallet):
            raise HTTPException(status_code=400, detail="Invalid Ethereum address")
        tokens = await eth_trace_wallet(wallet)
    else:
        raise HTTPException(status_code=400, detail="Unsupported chain")

    # Placeholder scoring logic
    score = 0
    reasons = []

    if not tokens:
        score = 90
        reasons.append("No tokens found — likely inactive or empty wallet.")
    else:
        score = 30
        reasons.append("Wallet holds tokens — further analysis required.")

    top_token = tokens[0] if tokens else TokenHolding(mint="unknown", amount=0, decimals=0, ui_amount=0.0)

    metadata = Metadata(
        mint=top_token.mint,
        owner=wallet,
        token_amount=top_token.amount,
        decimals=top_token.decimals,
        supply=None,
        name=top_token.name,
        symbol=top_token.symbol,
        logo_uri=top_token.logo_uri,
    )

    return AnalyzeResult(
        address=wallet,
        chain=chain,
        metadata=metadata,
        risk_score=RiskScore(score=score, reasons=reasons),
    )

