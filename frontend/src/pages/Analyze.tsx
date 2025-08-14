import React, { useState } from "react";
import TokenCard from "../components/TokenCard";

const Analyze = () => {
  const [tokens, setTokens] = useState([]);
  const [loading, setLoading] = useState(false);
  const [address, setAddress] = useState("");

  const handleAnalyze = async () => {
    setLoading(true);
    try {
      const res = await fetch(`/analyze?address=${address}&chain=solana`);
      const { result } = await res.json();
      setTokens(result || []);
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
        placeholder="Enter Solana address"
      />
      <button onClick={handleAnalyze} disabled={loading}>
        {loading ? "Analyzing..." : "Analyze"}
      </button>

      {tokens.map((token, idx) => (
        <TokenCard key={idx} token={token} />
      ))}
    </div>
  );
};

export default Analyze;

