iimport React from 'react';
import './TokenCard.css'; // Assuming CSS is modular and themed

interface TokenCardProps {
  name: string;
  symbol: string;
  amount: number;
  rawAmount: string;
  mint: string;
  icon?: string;
}

const TokenCard: React.FC<TokenCardProps> = ({
  name,
  symbol,
  amount,
  rawAmount,
  mint,
  icon
}) => {
  return (
    <div className="token-card">
      <div className="token-header">
        <img
          src={icon || '/placeholder.png'}
          alt={symbol}
          className="token-icon"
        />
        <h2 className="token-title">
          {name} ({symbol})
        </h2>
      </div>
      <div className="token-body">
        <p className="token-amount">
          {amount.toLocaleString()} {symbol}
        </p>
        <p className="token-raw">
          Raw: <code>{rawAmount}</code>
        </p>
        <p className="token-mint">
          Mint: <code>{mint}</code>
        </p>
      </div>
    </div>
  );
};

export default TokenCard;
