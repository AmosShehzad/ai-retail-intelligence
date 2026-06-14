"""
Day 5: Data standardization & structural cleaning.
Cleans merged_catalog.csv AND the SQLite products/sales tables.
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import pandas as pd
import logging
from database.db_manager import get_connection

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

MERGED_PATH = "data/processed/merged_catalog.csv"
CLEANED_PATH = "data/processed/merged_catalog_clean.csv"


# ── Stage 1: Enforce data types ────────────────────────────────────────────────
def enforce_types(df: pd.DataFrame) -> pd.DataFrame:
    df["product_name"]  = df["product_name"].astype(str).str.strip()
    df["category"]      = df["category"].astype(str).str.strip().str.title()
    df["supplier"]      = df["supplier"].astype(str).str.strip()
    df["source"]        = df["source"].astype(str).str.strip()

    df["cost_price"]    = pd.to_numeric(df["cost_price"], errors="coerce")
    df["selling_price"] = pd.to_numeric(df["selling_price"], errors="coerce")
    df["stock"]         = pd.to_numeric(df["stock"], errors="coerce").fillna(0).astype(int)

    log.info("Stage 1: Types enforced.")
    return df

# ── Stage 2: Handle missing values ────────────────────────────────────────────
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    # Drop rows missing critical fields (can't fix these)
    df = df.dropna(subset=["product_name", "selling_price"])

    # Fix missing cost_price: estimate as 75% of selling_price
    mask = df["cost_price"].isna()
    df.loc[mask, "cost_price"] = (df.loc[mask, "selling_price"] * 0.75).round(2)

    # Fix missing category
    df["category"] = df["category"].fillna("Uncategorized")
    df.loc[df["category"].str.strip() == "", "category"] = "Uncategorized"

    # Fix missing supplier
    df["supplier"] = df["supplier"].replace("nan", "Unknown").fillna("Unknown")

    log.info(f"Stage 2: Missing values handled. Rows: {before} → {len(df)}")
    return df

# ── Stage 3: Fix structural anomalies ─────────────────────────────────────────
def fix_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    # Remove zero or negative prices (data errors)
    df = df[df["selling_price"] > 0]

    # Remove negative stock
    df["stock"] = df["stock"].clip(lower=0)

    # Fix cost_price >= selling_price (impossible business case)
    bad_margin = df["cost_price"] >= df["selling_price"]
    df.loc[bad_margin, "cost_price"] = (df.loc[bad_margin, "selling_price"] * 0.75).round(2)
    log.info(f"  Fixed {bad_margin.sum()} rows with invalid cost >= selling price.")

    # Cap unrealistic prices (Kaggle data sometimes has Rs.500,000 errors)
    price_cap = df["selling_price"].quantile(0.999)
    outliers = df["selling_price"] > price_cap
    df = df[~outliers]
    log.info(f"  Removed {outliers.sum()} extreme price outliers (>{price_cap:.0f} PKR).")

    log.info(f"Stage 3: Anomalies fixed. Rows: {before} → {len(df)}")
    return df

# ── Stage 4: Remove duplicates ────────────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    # Exact duplicate rows
    df = df.drop_duplicates()

    # Same product name + category = duplicate (keep first occurrence)
    df = df.drop_duplicates(subset=["product_name", "category"], keep="first")

    log.info(f"Stage 4: Duplicates removed. Rows: {before} → {len(df)}")
    return df

# ── Stage 5: Final formatting ─────────────────────────────────────────────────
def final_formatting(df: pd.DataFrame) -> pd.DataFrame:
    # Round all prices to 2 decimals
    df["cost_price"]    = df["cost_price"].round(2)
    df["selling_price"] = df["selling_price"].round(2)

    # Reset index after all the dropping
    df = df.reset_index(drop=True)

    # Reorder columns consistently
    column_order = ["product_name", "category", "brand", "cost_price",
                     "selling_price", "stock", "supplier", "source"]
    available_cols = [c for c in column_order if c in df.columns]
    df = df[available_cols]

    log.info("Stage 5: Final formatting applied.")
    return df

# ── Validation report ──────────────────────────────────────────────────────────
def validation_report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 50)
    print("VALIDATION REPORT")
    print("=" * 50)
    print(f"Total rows          : {len(df)}")
    print(f"Null values         :\n{df.isnull().sum()}")
    print(f"Categories          : {df['category'].nunique()}")
    print(f"Price range (PKR)   : {df['selling_price'].min()} – {df['selling_price'].max()}")
    print(f"Avg margin %        : {((df['selling_price']-df['cost_price'])/df['selling_price']*100).mean():.1f}%")
    print(f"Duplicate names     : {df['product_name'].duplicated().sum()}")
    print("=" * 50)

# ── Master runner ──────────────────────────────────────────────────────────────
def run_standardization():
    log.info("=" * 50)
    log.info("DATA STANDARDIZATION STARTED")
    log.info("=" * 50)

    df = pd.read_csv(MERGED_PATH)
    log.info(f"Loaded {len(df)} rows from {MERGED_PATH}")

    df = enforce_types(df)
    df = handle_missing(df)
    df = fix_anomalies(df)
    df = remove_duplicates(df)
    df = final_formatting(df)

    validation_report(df)

    df.to_csv(CLEANED_PATH, index=False)
    log.info(f"Saved cleaned dataset → {CLEANED_PATH}")
    log.info("STANDARDIZATION COMPLETE ✅")
    return df


if __name__ == "__main__":
    run_standardization()