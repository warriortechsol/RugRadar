import React, { useState } from "react";
import TokenCard from "../components/TokenCard";

type AnalyzeResult = {
  address: string;
  chain: string;
  metadata?: {
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
  events?: { type: string; pct: number; timestamp: string }[];
  holders?: { address: string; amount: number; pct: number }[];
};

const Analyze = () => {
  const [analysis, setAnalysis] = useState<AnalyzeResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [address, setAddress] = useState("");

  const handleAnalyze = async () => {
    if (!address.trim()) return;
    setLoading(true);
    try {
      const res = await fetch(
        `${import.meta.env.VITE_API_BASE}analyze?mint=${address.trim()}&chain=solana`
      );
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || "Analyze failed");
      setAnalysis(data);
    } catch (err) {
      console.error("Fetch failed:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="analyze-page">
      <input
        type="text"
        value={address}
        onChange={(e) => setAddress(e.target.value)}
        placeholder="Enter Solana mint address"
      />
      <button onClick={handleAnalyze} disabled={loading}>
        {loading ? "Analyzing..." : "Analyze"}
      </button>

      {analysis && (
        <div className="analysis-card" style={{ marginTop: 16 }}>
          <h3>
            {analysis.metadata?.symbol || "Token"} —{" "}
            {analysis.metadata?.name || analysis.metadata?.mint}
          </h3>
          <p>
            Supply: {analysis.metadata?.supply ?? "—"} • Decimals:{" "}
            {analysis.metadata?.decimals ?? "—"}
          </p>
          <p>
            Risk score: {analysis.risk_score.score} / 100
          </p>
          <ul>
            {analysis.risk_score.reasons.length
              ? analysis.risk_score.reasons.map((r, i) => <li key={i}>{r}</li>)
              : <li>No issues detected</li>}
          </ul>

          {analysis.holders && analysis.holders.length > 0 && (
            <div className="top-holders" style={{ marginTop: 12 }}>
              <strong>Top Holders:</strong>
              <table style={{ width: "100%", marginTop: 8, fontSize: 14 }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: "left" }}>Address</th>
                    <th style={{ textAlign: "right" }}>Amount</th>
                    <th style={{ textAlign: "right" }}>% of Supply</th>
                  </tr>
                </thead>
                <tbody>
                  {analysis.holders.map((h, i) => (
                    <tr key={i}>
                      <td style={{ fontFamily: "monospace" }}>{h.address}</td>
                      <td style={{ textAlign: "right" }}>
                        {h.amount.toLocaleString()}
                      </td>
                      <td style={{ textAlign: "right" }}>
                        {h.pct.toFixed(4)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default Analyze;
