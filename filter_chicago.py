"""
One-time script: reads the full itineraries.csv in chunks and writes
only Chicago (ORD/MDW) flights to data/chicago_itineraries.csv.
Run once, then preprocess.py uses the smaller file automatically.
"""
import os
import pandas as pd
import kagglehub

_dataset_path = kagglehub.dataset_download("dilwong/flightprices")
CSV_PATH = os.path.join(_dataset_path, "itineraries.csv")
OUT_PATH = "data/chicago_itineraries.csv"

os.makedirs("data", exist_ok=True)

CHICAGO = {"ORD", "MDW"}
CHUNK_SIZE = 100_000

print(f"Reading {CSV_PATH} in chunks of {CHUNK_SIZE:,}...")
total_read = 0
total_kept = 0
first_chunk = True

with pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE) as reader:
    for chunk in reader:
        total_read += len(chunk)
        mask = chunk["startingAirport"].isin(CHICAGO) | chunk["destinationAirport"].isin(CHICAGO)
        filtered = chunk[mask]
        total_kept += len(filtered)
        filtered.to_csv(OUT_PATH, mode="w" if first_chunk else "a",
                        header=first_chunk, index=False)
        first_chunk = False
        print(f"  Read {total_read:,} rows, kept {total_kept:,} Chicago rows...", end="\r")

print(f"\nDone. Kept {total_kept:,} / {total_read:,} rows ({total_kept/total_read:.1%})")
print(f"Saved to {OUT_PATH}")
