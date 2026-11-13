"""Clean and pseudonymize the TellCo xDR CSV for dashboard upload.

Usage:
  python scripts/clean_xdr.py input.csv tellco_xdr_deployment_anonymized.csv
"""
from __future__ import annotations
import hashlib
import re
import secrets
import sys
from pathlib import Path
import pandas as pd

def snake(name: str) -> str:
    name = name.strip().replace("<", " lt ").replace(">", " gt ").replace("/", " ")
    name = re.sub(r"[()]", "", name)
    name = re.sub(r"[^A-Za-z0-9]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_").lower()

def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python scripts/clean_xdr.py input.csv output.csv")
    source, destination = map(Path, sys.argv[1:])
    frame = pd.read_csv(source, dtype=str, low_memory=False)
    frame.columns = [snake(c) for c in frame.columns]
    ids = {"bearer_id", "imsi", "msisdn_number", "imei"}
    categories = {"start", "end", "last_location_name", "handset_manufacturer", "handset_type"}
    for col in frame.columns:
        if col in ids or col in categories:
            frame[col] = frame[col].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        else:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in ("start", "end"):
        frame[col] = pd.to_datetime(frame[col], format="%m/%d/%Y %H:%M", errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S").astype("string")
    frame = frame.drop_duplicates()
    if "msisdn_number" not in frame:
        raise ValueError("Required customer identifier MSISDN/Number was not found.")
    frame = frame[frame.msisdn_number.notna()].copy()
    salt = secrets.token_bytes(32)
    frame.insert(0, "customer_id", frame.msisdn_number.map(lambda value: hashlib.sha256(salt + str(value).encode()).hexdigest()[:16]))
    for col in ("last_location_name", "handset_manufacturer", "handset_type"):
        if col in frame:
            frame[col] = frame[col].fillna("Unknown")
    frame["duration_s"] = frame["dur_ms"] / 1000 if "dur_ms" in frame else 0
    if {"total_dl_bytes", "total_ul_bytes"}.issubset(frame.columns):
        frame["total_traffic_bytes"] = frame[["total_dl_bytes", "total_ul_bytes"]].sum(axis=1, min_count=1)
    frame = frame.drop(columns=[c for c in ("msisdn_number", "imsi", "imei", "bearer_id", "last_location_name") if c in frame])
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f"Wrote {len(frame):,} rows and {len(frame.columns)} columns to {destination}")

if __name__ == "__main__":
    main()
