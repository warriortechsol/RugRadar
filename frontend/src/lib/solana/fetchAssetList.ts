import { Connection, PublicKey } from "@solana/web3.js";

const connection = new Connection("https://api.mainnet-beta.solana.com");

export async function fetchAssetList(walletAddress: string) {
  const owner = new PublicKey(walletAddress);
  const accounts = await connection.getParsedTokenAccountsByOwner(owner, {
    programId: new PublicKey("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
  });

  const holdings = accounts.value.map(({ account }) => {
    const info = account.data.parsed.info;
    const amount = parseFloat(info.tokenAmount.uiAmount || "0");
    return {
      mint: info.mint,
      amount,
    };
  });

  return holdings;
}

