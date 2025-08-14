import { Connection, PublicKey } from "@solana/web3.js";
import {
  Metadata,
  PROGRAM_ID as METADATA_PROGRAM_ID
} from "@metaplex-foundation/mpl-token-metadata";

const SOLANA_RPC = "https://api.mainnet-beta.solana.com";
const connection = new Connection(SOLANA_RPC);

export async function fetchTokenMetadata(mintAddress: string) {
  try {
    const mintPublicKey = new PublicKey(mintAddress);
    const metadataPDA = await Metadata.getPDA(mintPublicKey);
    const metadata = await Metadata.load(connection, metadataPDA);
    const { name, symbol, uri } = metadata.data.data;

    return {
      name,
      symbol,
      mint: mintAddress,
      icon: uri,
    };
  } catch {
    return {
      name: "Unknown",
      symbol: "",
      mint: mintAddress,
      icon: "",
    };
  }
}

