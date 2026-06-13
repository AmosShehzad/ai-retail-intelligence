"""Online Retail II ingestion pipeline."""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from data.schema_mapping import FILTERS, GBP_TO_PKR, KAGGLE_TO_STANDARD, OUTPUT_COLUMNS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

RAW_PATH = BASE_DIR / "data" / "raw" / "online_retail_II.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "retail_cleaned.csv"


def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    log.info("Loading raw data from: %s", path)
    df = pd.read_csv(path, dtype={"Customer ID": str})
    log.info("Loaded %s raw rows", len(df))
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=KAGGLE_TO_STANDARD)
    log.info("Columns renamed to standard schema")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    original_len = len(df)

    df = df.dropna(subset=["product_name", "unit_price_gbp"])
    df = df[df["quantity"] >= FILTERS["min_quantity"]]
    df = df[df["unit_price_gbp"] >= FILTERS["min_unit_price"]]
    df = df[df["product_code"].str.match(r"^\d", na=False)]
    df["product_name"] = df["product_name"].str.strip().str.title()
    df["sale_date"] = pd.to_datetime(df["sale_date"]).dt.date

    log.info("Cleaned: %s → %s rows", original_len, len(df))
    return df


def categorize_product(name: str) -> str:
    name = name.lower()
    rules = {
        "Tea & Beverages": ["tea", "coffee", "juice", "drink", "water", "cup"],
        "Snacks & Food": ["cake", "biscuit", "chocolate", "sweet", "snack"],
        "Home & Kitchen": ["glass", "bowl", "plate", "kitchen", "jar"],
        "Clothing": ["bag", "shirt", "dress"],
        "Decoration": ["candle", "light", "frame", "flower"],
        "Stationery": ["pen", "book", "card"],
        "Personal Care": ["soap", "cream", "lotion"],
    }

    for category, keywords in rules.items():
        if any(keyword in name for keyword in keywords):
            return category

    return "General"


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    df["unit_price_pkr"] = (df["unit_price_gbp"] * GBP_TO_PKR).round(2)
    df["category"] = df["product_name"].apply(categorize_product)
    log.info("Transformation complete")
    return df


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [column for column in OUTPUT_COLUMNS if column in df.columns]
    return df[cols]


def save_processed(df: pd.DataFrame) -> None:
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    log.info("Saved processed dataset to %s", PROCESSED_PATH)


def load_into_database(df: pd.DataFrame) -> None:
    import database.db_manager as db_manager

    db_manager = importlib.reload(db_manager)
    db_manager.initialize_database()

    conn = db_manager.get_connection()
    cursor = conn.cursor()

    inserted_products = 0
    inserted_sales = 0

    for _, row in df.iterrows():
        cursor.execute(
            "SELECT product_id FROM products WHERE product_name = ?",
            (row["product_name"],),
        )
        result = cursor.fetchone()

        if result is None:
            cursor.execute(
                """
                INSERT INTO products
                (product_name, category, cost_price, selling_price, stock)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    row["product_name"],
                    row["category"],
                    round(row["unit_price_pkr"] * 0.75, 2),
                    row["unit_price_pkr"],
                    0,
                ),
            )
            product_id = cursor.lastrowid
            inserted_products += 1
        else:
            product_id = result[0]

        cursor.execute(
            "INSERT INTO sales (product_id, quantity, sale_date) VALUES (?, ?, ?)",
            (product_id, int(row["quantity"]), str(row["sale_date"])),
        )
        inserted_sales += 1

    conn.commit()
    conn.close()
    log.info("DB insert complete: %s new products, %s sales", inserted_products, inserted_sales)


def run_pipeline(load_db: bool = True) -> pd.DataFrame:
    log.info("%s", "=" * 50)
    log.info("KAGGLE PIPELINE STARTED")
    log.info("%s", "=" * 50)

    df = load_raw_data()
    df = rename_columns(df)
    df = clean_data(df)
    df = transform_data(df)
    df = select_output_columns(df)
    save_processed(df)

    if load_db:
        load_into_database(df.head(5000))

    log.info("PIPELINE COMPLETE")
    return df


if __name__ == "__main__":
    run_pipeline(load_db=True)
