from __future__ import annotations

import re

from app.storage.sqlite_store import get_store

# Canonical mapping: common user-facing names → ticker stored in mock DB
_ASSET_ALIASES: dict[str, str] = {
    "bitcoin": "BTC",
    "btc": "BTC",
    "ethereum": "ETH",
    "ether": "ETH",
    "eth": "ETH",
    "usdc": "USDC",
    "usd coin": "USDC",
    "solana": "SOL",
    "sol": "SOL",
    "litecoin": "LTC",
    "ltc": "LTC",
    "dogecoin": "DOGE",
    "doge": "DOGE",
    "xrp": "XRP",
    "ripple": "XRP",
    "matic": "MATIC",
    "polygon": "MATIC",
}


def normalize_asset(asset: str) -> str:
    """Return canonical uppercase ticker, or original uppercased if unknown."""
    a = asset.strip()
    return _ASSET_ALIASES.get(a.lower(), a.upper())


def validate_tx_id(tx: str) -> bool:
    t = tx.strip()
    return bool(re.match(r"^[A-Za-z0-9\-]{6,64}$", t))


def validate_asset(asset: str) -> bool:
    a = asset.strip()
    return 2 <= len(a) <= 12


def check_transaction(transaction_id: str, asset_type: str) -> dict:
    if not validate_tx_id(transaction_id):
        return {
            "ok": False,
            "error": "invalid_transaction_id",
            "message": "Please provide a valid transaction ID (letters, numbers, dashes only).",
        }
    if not validate_asset(asset_type):
        return {
            "ok": False,
            "error": "invalid_asset",
            "message": "Please provide a short asset symbol like BTC, ETH, or USDC.",
        }
    canonical = normalize_asset(asset_type)
    row = get_store().lookup_transaction(transaction_id, canonical)
    if not row:
        # Fallback: try with the raw asset string the user provided
        row = get_store().lookup_transaction(transaction_id, asset_type)
    if not row:
        return {
            "ok": True,
            "found": False,
            "message": (
                "I couldn't find that transaction in our mock ledger. "
                "Please double-check the transaction ID and asset type, "
                "or open a support ticket for further investigation."
            ),
        }
    return {
        "ok": True,
        "found": True,
        "transaction_id": row["tx_id"],
        "asset_type": row["asset_type"],
        "status": row["status"],
        "detail": row["detail"],
        "next_steps": [
            "If status is **pending**, wait for network confirmations.",
            "If status is **failed**, verify the destination address and selected network.",
            "If status is **delayed review**, watch for a follow-up email from Coinbase support.",
        ],
    }
