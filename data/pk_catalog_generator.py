"""
Day 4: Synthetic Pakistani product catalog generator.
Generates realistic kiryana store products with logical pricing.
Output: data/processed/pk_catalog.csv
"""

import sys
from pathlib import Path
import pandas as pd
import random
import os
import logging

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from data.pk_catalog_config import CATALOG, STOCK_RANGES, SIZE_MULTIPLIERS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

OUTPUT_PATH = BASE_DIR / "data" / "processed" / "pk_catalog.csv"
random.seed(42)  # reproducible results


def calculate_price(base_price: float, variant: str, noise: float = 0.08) -> tuple[float, float]:
    """
    Calculates selling_price and cost_price for a product variant.

    Logic:
    - base_price × size_multiplier = raw price
    - Add ±8% random noise (simulates market variation)
    - Round to nearest 5 PKR (retail convention)
    - cost_price = selling_price × category margin
    """
    multiplier   = SIZE_MULTIPLIERS.get(variant, 1.0)
    raw_price    = base_price * multiplier
    noisy_price  = raw_price * random.uniform(1 - noise, 1 + noise)
    selling_price = round(noisy_price / 5) * 5  # round to nearest 5 PKR
    selling_price = max(selling_price, 10)       # floor at Rs.10
    return selling_price


def generate_catalog() -> pd.DataFrame:
    records = []
    product_id = 1

    for category, config in CATALOG.items():
        brands       = config["brands"]
        products     = config["products"]
        cost_margin  = config["cost_margin"]
        stock_min, stock_max = STOCK_RANGES[category]

        for brand in brands:
            for product in products:
                for variant in product["variants"]:

                    selling_price = calculate_price(product["base_price"], variant)
                    cost_price    = round(selling_price * cost_margin / 5) * 5
                    cost_price    = max(cost_price, 5)

                    # Logical check: cost must always be less than selling price
                    assert cost_price < selling_price, (
                        f"Pricing error: {brand} {product['name']} {variant}"
                    )

                    stock = random.randint(stock_min, stock_max)

                    records.append({
                        "product_id"   : product_id,
                        "product_name" : f"{brand} {product['name']} {variant}",
                        "brand"        : brand,
                        "category"     : category,
                        "variant"      : variant,
                        "cost_price"   : cost_price,
                        "selling_price": selling_price,
                        "stock"        : stock,
                        "supplier"     : f"{brand} Pakistan",
                    })
                    product_id += 1

    df = pd.DataFrame(records)
    log.info(f"Generated {len(df)} Pakistani products across "
             f"{df['category'].nunique()} categories.")
    return df


def validate_catalog(df: pd.DataFrame) -> None:
    """Runs sanity checks on the generated catalog."""
    errors = []

    if (df["cost_price"] >= df["selling_price"]).any():
        errors.append("❌ Some products have cost >= selling price")

    if df["product_name"].duplicated().any():
        errors.append(f"❌ {df['product_name'].duplicated().sum()} duplicate product names")

    if (df["stock"] < 0).any():
        errors.append("❌ Negative stock found")

    if errors:
        for e in errors: log.error(e)
        raise ValueError("Catalog validation failed.")

    log.info(f"✅ Validation passed: {len(df)} products, "
             f"avg margin = {((df['selling_price'] - df['cost_price']) / df['selling_price'] * 100).mean():.1f}%")


def save_catalog(df: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    log.info(f"Saved catalog → {OUTPUT_PATH}")


def load_catalog_into_db(df: pd.DataFrame) -> None:
    """Inserts generated catalog into SQLite products table."""
    from database.db_manager import get_connection

    conn   = get_connection()
    cursor = conn.cursor()

    # Only insert products not already in DB (avoid duplicates with Day 2 seed)
    cursor.execute("SELECT product_name FROM products")
    existing = {row[0] for row in cursor.fetchall()}

    new_products = df[~df["product_name"].isin(existing)]

    if new_products.empty:
        log.info("ℹ️  All catalog products already in DB. Skipping insert.")
        conn.close()
        return

    records = [
        (
            row["product_name"],
            row["category"],
            row["cost_price"],
            row["selling_price"],
            row["stock"],
            row["supplier"],
        )
        for _, row in new_products.iterrows()
    ]

    cursor.executemany(
        """INSERT INTO products
           (product_name, category, cost_price, selling_price, stock, supplier)
           VALUES (?, ?, ?, ?, ?, ?)""",
        records
    )

    conn.commit()
    conn.close()
    log.info(f"✅ Inserted {len(records)} new Pakistani products into DB.")


def run_generator(load_db: bool = True) -> pd.DataFrame:
    log.info("=" * 50)
    log.info("PK CATALOG GENERATOR STARTED")
    log.info("=" * 50)

    df = generate_catalog()
    validate_catalog(df)
    save_catalog(df)

    if load_db:
        load_catalog_into_db(df)

    # Print summary table
    summary = df.groupby("category").agg(
        products    = ("product_id", "count"),
        avg_price   = ("selling_price", "mean"),
        avg_margin  = ("selling_price", lambda x: (
            (x - df.loc[x.index, "cost_price"]).mean() / x.mean() * 100
        ))
    ).round(1)
    print("\n" + summary.to_string())

    log.info("GENERATOR COMPLETE ✅")
    return df


if __name__ == "__main__":
    run_generator(load_db=True)