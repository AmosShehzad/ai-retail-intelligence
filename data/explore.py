"""Quick EDA on raw Kaggle data — run once to understand the dataset."""

import pandas as pd
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parent / "raw" / "online_retail_II.csv"

df = pd.read_csv(RAW_PATH, nrows=10000)

print("\n── Shape ──────────────────────────")
print(df.shape)

print("\n── Columns & Types ────────────────")
print(df.dtypes)

print("\n── Missing Values ─────────────────")
print(df.isnull().sum())

print("\n── Sample Rows ────────────────────")
print(df.head(3).to_string())

print("\n── Numeric Summary ────────────────")
print(df[["Quantity", "Price"]].describe())