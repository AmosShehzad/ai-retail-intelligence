"""
Day 5: Cleans structural anomalies directly inside the SQLite database.
Run AFTER standardize.py, since DB already has Day 2-4 data inserted.
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import logging
from database.db_manager import get_connection

# Setup logging to track what the script is doing
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


# -----------------------------
# FIX DATA ISSUES IN PRODUCTS TABLE
# -----------------------------
def clean_products_table():
    # Connect to SQLite database
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Fix invalid cost_price (cost must always be less than selling price)
    cursor.execute("""
        UPDATE products
        SET cost_price = ROUND(selling_price * 0.75, 2)
        WHERE cost_price >= selling_price OR cost_price <= 0
    """)
    log.info(f"Fixed {cursor.rowcount} rows: invalid cost_price.")

    # 2. Fix negative stock values (not valid in real inventory)
    cursor.execute("""
        UPDATE products
        SET stock = 0
        WHERE stock < 0
    """)
    log.info(f"Fixed {cursor.rowcount} rows: negative stock.")

    # 3. Fix missing or empty category values
    cursor.execute("""
        UPDATE products
        SET category = 'Uncategorized'
        WHERE category IS NULL OR TRIM(category) = ''
    """)
    log.info(f"Fixed {cursor.rowcount} rows: missing category.")

    # 4. Remove invalid products with zero or negative price
    cursor.execute("""
        DELETE FROM products
        WHERE selling_price <= 0
    """)
    log.info(f"Removed {cursor.rowcount} rows: zero/negative price.")

    # Save changes to database
    conn.commit()
    conn.close()


# -----------------------------
# FIX ORPHAN SALES DATA
# -----------------------------
def remove_orphan_sales():
    """
    Deletes sales records that reference deleted products.
    (Prevents broken foreign key relationships)
    """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM sales
        WHERE product_id NOT IN (SELECT product_id FROM products)
    """)
    log.info(f"Removed {cursor.rowcount} orphan sales records.")

    conn.commit()
    conn.close()


# -----------------------------
# REMOVE DUPLICATE PRODUCTS
# -----------------------------
def remove_duplicate_products():
    """
    Keeps only one record per (product_name, category).
    Removes duplicates based on smallest product_id.
    """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM products
        WHERE product_id NOT IN (
            SELECT MIN(product_id)
            FROM products
            GROUP BY product_name, category
        )
    """)
    log.info(f"Removed {cursor.rowcount} duplicate products.")

    conn.commit()
    conn.close()


# -----------------------------
# MAIN DATABASE CLEANING PIPELINE
# -----------------------------
def run_db_cleaning():
    log.info("=" * 50)
    log.info("DATABASE CLEANING STARTED")
    log.info("=" * 50)

    # Step 1: Remove duplicates first
    remove_duplicate_products()

    # Step 2: Fix invalid product data
    clean_products_table()

    # Step 3: Clean sales table consistency
    remove_orphan_sales()

    log.info("DATABASE CLEANING COMPLETE ✅")


# Run script directly
if __name__ == "__main__":
    run_db_cleaning()