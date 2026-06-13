"""
Merges Kaggle processed data + PK synthetic catalog into a unified dataset.
This is what your analytics engine and RAG pipeline will read.
"""

import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

KAGGLE_PATH = "data/processed/retail_cleaned.csv"
CATALOG_PATH = "data/processed/pk_catalog.csv"
MERGED_PATH  = "data/processed/merged_catalog.csv"


def merge_datasets() -> pd.DataFrame:
    kaggle  = pd.read_csv(KAGGLE_PATH)
    catalog = pd.read_csv(CATALOG_PATH)

    # Normalize column names for merge compatibility
    kaggle_normalized = pd.DataFrame({
        "product_name" : kaggle["product_name"],
        "category"     : kaggle["category"],
        "selling_price": kaggle["unit_price_pkr"],
        "cost_price"   : (kaggle["unit_price_pkr"] * 0.75).round(2),
        "stock"        : 0,
        "supplier"     : "Kaggle Import",
        "source"       : "kaggle",
    })

    catalog_normalized = catalog[[
        "product_name", "category", "cost_price",
        "selling_price", "stock", "supplier"
    ]].copy()
    catalog_normalized["source"] = "pk_synthetic"

    merged = pd.concat([catalog_normalized, kaggle_normalized], ignore_index=True)
    merged = merged.drop_duplicates(subset=["product_name"])

    os.makedirs("data/processed", exist_ok=True)
    merged.to_csv(MERGED_PATH, index=False)

    log.info(f"Merged dataset: {len(catalog_normalized)} PK products + "
             f"{len(kaggle_normalized)} Kaggle products = {len(merged)} total")
    log.info(f"Saved → {MERGED_PATH}")
    return merged


if __name__ == "__main__":
    merge_datasets()