import React, { useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

type TokenHolding = {
  mint: string;
  amount: number;
  decimals: number;
  ui_amount: number;
  name?: string | null;
  symbol?: string | null;
  logo_uri?: string | null;
};

type TraceResult = {
  wallet: string;
  chain: string;
  tokens: TokenHolding[];
};

type AnalyzeResult = {
  address: string;
  chain: string;
  metadata: {
    mint: string;
    owner?: string | null;
    token_amount?: number | null;
    decimals?: number | null;
    supply?: number | null;
    name?: string | null;
    symbol?: string | null;
    logo_uri?: string | null;
  };
  risk_score: { score: number; reasons: string[] };
};

export default function App() {
  const [address, setAddress] = useState("");
  const [chain, setChain] = useState<"solana" | "ethereum">("solana");
  const [loading, setLoading] = useState<"idle" | "trace" | "analyze">("idle");
  const [trace, setTrace] = useState<TraceResult | null>(null);
  const [analyze, setAnalyze] = useState<AnalyzeResult | null>(null);
  const valid = useMemo(() => address.trim().length > 0, [address]);

  async function runTrace() {
    if (!valid) return;
    setLoading("trace");
    setAnalyze(null);
    try {
      const q = new URLSearchParams({ wallet: address.trim(), chain });
      const res = await fetch(`${API_BASE}/trace?` + q.toString());
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || "Trace failed");
      setTrace(data);
    } catch (e: any) {
      alert(e.message || "Trace failed");
    } finally {
      setLoading("idle");
    }
  }

  async function runAnalyze() {
    if (!valid) return;
    setLoading("analyze");
    setTrace(null);
    try {
      const q = new URLSearchParams({ address: address.trim(), chain });
      const res = await fetch(`${API_BASE}/analyze?` + q.toString());
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || "Analyze failed");
      setAnalyze(data);
    } catch (e: any) {
      alert(e.message || "Analyze failed");
    } finally {
      setLoading("idle");
    }
  }

  return (
    <div style={{ maxWidth: 1080, margin: "40px auto", padding: "0 16px" }}>
      <h1 style={{ marginBottom: 16 }}>RugRadar</h1>
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="row" style={{ gap: 12, flexWrap: "wrap" }}>
          <input
            style={{ flex: 1, minWidth: 280 }}
            placeholder={chain === "solana" ? "Solana wallet or mint" : "Ethereum address or contract"}
            value={address}
            onChange={(e) => setAddress(e.target.value)}
          />
          <select value={chain} onChange={(e) => setChain(e.target.value as any)}>
            <option value="solana">Solana</option>
            <option value="ethereum">Ethereum</option>
          </select>
          <button onClick={runTrace} disabled={!valid || loading !== "idle"}>
            {loading === "trace" ? "Tracing..." : "Trace"}
          </button>
          <button onClick={runAnalyze} disabled={!valid || loading !== "idle"}>
            {loading === "analyze" ? "Analyzing..." : "Analyze"}
          </button>
        </div>
        <p style={{ opacity: 0.7, marginTop: 8 }}>
          Address: <code>{address || "—"}</code> • Chain: <code>{chain}</code>
        </p>
      </div>

      {trace && (
        <div className="card">
          <h3 style={{ marginTop: 0, marginBottom: 12 }}>Holdings</h3>
          <div className="grid">
            {trace.tokens.map((t) => (
              <div key={t.mint} className="card" style={{ padding: 12 }}>
                <div className="row" style={{ alignItems: "center" }}>
                  {t.logo_uri ? (
                    <img src={t.logo_uri} alt="" width={24} height={24} style={{ borderRadius: 6, marginRight: 8 }} />
                  ) : (
                    <div style={{ width: 24, height: 24, background: "#1b2634", borderRadius: 6, marginRight: 8 }} />
                  )}
                  <strong>{t.symbol || "Unknown"}</strong>
                  <span style={{ opacity: 0.7, marginLeft: 6 }}>{t.name || t.mint}</span>
                </div>
                <div style={{ marginTop: 8, opacity: 0.85 }}>
                  {t.ui_amount.toLocaleString(undefined, { maximumFractionDigits: 6 })} ({t.amount} raw)
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {analyze && (
        <div className="card">
          <h3 style={{ marginTop: 0, marginBottom: 12 }}>Analysis</h3>
          <div className="row" style={{ gap: 12, alignItems: "center" }}>
            {analyze.metadata.logo_uri ? (
              <img src={analyze.metadata.logo_uri} width={28} height={28} style={{ borderRadius: 8 }} />
            ) : (
              <div style={{ width: 28, height: 28, background: "#1b2634", borderRadius: 8 }} />
            )}
            <div>
              <div><strong>{analyze.metadata.symbol || "Token"}</strong> — {analyze.metadata.name || analyze.metadata.mint}</div>
              <div style={{ opacity: 0.7, fontSize: 12 }}>
                Supply: {analyze.metadata.supply ?? "—"} • Decimals: {analyze.metadata.decimals ?? "—"}
              </div>
            </div>
          </div>
          <div className="card" style={{ marginTop: 12 }}>
            <strong>Risk score:</strong> {analyze.risk_score.score} / 100
            <ul style={{ marginTop: 8 }}>
              {analyze.risk_score.reasons.length ? (
                analyze.risk_score.reasons.map((r, i) => <li key={i}>{r}</li>)
              ) : (
                <li>No issues detected in current heuristic</li>
              )}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

