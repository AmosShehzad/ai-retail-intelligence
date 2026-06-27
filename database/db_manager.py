"""
database/db_manager.py
Production schema for AI Retail Intelligence Assistant.
Four tables: products, sales, purchase_orders, query_logs
"""

import sqlite3
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_DEFAULT_DB_PATH = Path(__file__).resolve().parent / "retail.db"
_ENV_DB_PATH     = os.getenv("DB_PATH")
DB_PATH          = Path(_ENV_DB_PATH).expanduser() \
                   if _ENV_DB_PATH else _DEFAULT_DB_PATH

if not DB_PATH.is_absolute():
    DB_PATH = (Path(__file__).resolve().parent / DB_PATH).resolve()


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize_database():
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS products (
            product_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name        TEXT    NOT NULL UNIQUE,
            category            TEXT    NOT NULL,
            cost_price          REAL    NOT NULL CHECK(cost_price > 0),
            selling_price       REAL    NOT NULL CHECK(selling_price > cost_price),
            stock               INTEGER NOT NULL DEFAULT 0 CHECK(stock >= 0),
            supplier            TEXT,
            low_stock_threshold INTEGER NOT NULL DEFAULT 10,
            is_active           INTEGER NOT NULL DEFAULT 1,
            created_at          DATE    DEFAULT CURRENT_DATE,
            updated_at          DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS sales (
            sale_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            quantity   INTEGER NOT NULL CHECK(quantity > 0),
            sale_price REAL    NOT NULL CHECK(sale_price > 0),
            sale_date  DATE    NOT NULL,
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        );

        CREATE TABLE IF NOT EXISTS purchase_orders (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id       INTEGER NOT NULL,
            quantity_ordered INTEGER NOT NULL CHECK(quantity_ordered > 0),
            cost_per_unit    REAL    NOT NULL CHECK(cost_per_unit > 0),
            total_cost       REAL    NOT NULL CHECK(total_cost > 0),
            supplier         TEXT    NOT NULL,
            order_date       DATE    DEFAULT CURRENT_DATE,
            status           TEXT    NOT NULL DEFAULT 'pending'
                             CHECK(status IN ('pending','received','cancelled')),
            received_date    DATE,
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        );

        CREATE TABLE IF NOT EXISTS query_logs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            question  TEXT     NOT NULL,
            answer    TEXT     NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    conn.close()
    print("✅ Database initialized with new schema.")


if __name__ == "__main__":
    initialize_database()
print("=" * 60)
print("__file__       :", __file__)
print("_ENV_DB_PATH   :", _ENV_DB_PATH)
print("Resolved DBPATH:", DB_PATH)
print("Parent Folder  :", DB_PATH.parent)
print("=" * 60)