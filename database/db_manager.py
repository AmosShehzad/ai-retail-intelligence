import sqlite3
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

_DEFAULT_DB_PATH = Path(__file__).resolve().parent / "retail.db"
_ENV_DB_PATH = os.getenv("DB_PATH")
DB_PATH = Path(_ENV_DB_PATH).expanduser() if _ENV_DB_PATH else _DEFAULT_DB_PATH

if not DB_PATH.is_absolute():
    DB_PATH = (Path(__file__).resolve().parent / DB_PATH).resolve()


def get_connection():
    """Returns a connection to the SQLite database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets you access columns by name
    return conn


def initialize_database():
    """Creates all tables if they don't already exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS products (
            product_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name TEXT    NOT NULL,
            category     TEXT    NOT NULL,
            cost_price   REAL    NOT NULL,
            selling_price REAL   NOT NULL,
            stock        INTEGER NOT NULL DEFAULT 0,
            supplier     TEXT
        );

        CREATE TABLE IF NOT EXISTS sales (
            sale_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            quantity   INTEGER NOT NULL,
            sale_date  DATE    NOT NULL,
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        );

        CREATE TABLE IF NOT EXISTS competitor_prices (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name TEXT    NOT NULL,
            store_name   TEXT    NOT NULL,
            price        REAL    NOT NULL,
            scrape_date  DATE    NOT NULL
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
    print("✅ Database initialized successfully.")


def seed_pakistani_products():
    """
    Inserts realistic Pakistani kiryana store products.
    Only runs if the products table is empty.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM products")
    count = cursor.fetchone()[0]

    if count > 0:
        print("ℹ️  Products already seeded. Skipping.")
        conn.close()
        return

    products = [
        # (product_name, category, cost_price, selling_price, stock, supplier)
        ("Tapal Danedar 200g",      "Tea & Beverages",  180, 220, 80,  "Tapal Foods"),
        ("Tapal Family Mixture 95g","Tea & Beverages",  90,  115, 60,  "Tapal Foods"),
        ("Nescafe Classic 50g",     "Tea & Beverages",  320, 390, 30,  "Nestle Pakistan"),
        ("Surf Excel 500g",         "Detergents",       310, 380, 45,  "Unilever Pakistan"),
        ("Ariel 500g",              "Detergents",       290, 360, 30,  "P&G Pakistan"),
        ("Shan Biryani Masala 60g", "Spices & Masala",  75,  95,  100, "Shan Foods"),
        ("Shan Karahi Masala 50g",  "Spices & Masala",  65,  85,  90,  "Shan Foods"),
        ("National Achar 400g",     "Condiments",       160, 200, 40,  "National Foods"),
        ("Nestlé MilkPak 1L",       "Dairy",            170, 210, 55,  "Nestle Pakistan"),
        ("Olpers Milk 1L",          "Dairy",            165, 205, 60,  "Engro Foods"),
        ("Lux Soap 150g",           "Personal Care",    85,  110, 120, "Unilever Pakistan"),
        ("Safeguard Soap 175g",     "Personal Care",    95,  125, 100, "P&G Pakistan"),
        ("Dettol Antiseptic 60ml",  "Personal Care",    130, 165, 70,  "Reckitt Pakistan"),
        ("Colgate 75ml",            "Personal Care",    110, 145, 80,  "Colgate-Palmolive"),
        ("Knorr Noodles Chicken",   "Instant Food",     60,  80,  150, "Unilever Pakistan"),
        ("Indomie Noodles",         "Instant Food",     55,  75,  120, "Kolson"),
        ("Dalda Banaspati 1kg",     "Cooking Oil",      420, 490, 35,  "Dalda Foods"),
        ("Sufi Cooking Oil 1L",     "Cooking Oil",      400, 470, 40,  "Sufi Group"),
        ("Tang Orange 500g",        "Beverages",        180, 225, 50,  "Mondelez Pakistan"),
        ("Rooh Afza 800ml",         "Beverages",        350, 420, 25,  "Hamdard"),
    ]

    cursor.executemany(
        "INSERT INTO products (product_name, category, cost_price, selling_price, stock, supplier) VALUES (?,?,?,?,?,?)",
        products
    )

    conn.commit()
    conn.close()
    print(f"✅ Seeded {len(products)} Pakistani products.")


def seed_sample_sales():
    """
    Inserts 60 days of sample sales transactions.
    Only runs if the sales table is empty.
    """
    import random
    from datetime import date, timedelta

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM sales")
    if cursor.fetchone()[0] > 0:
        print("ℹ️  Sales already seeded. Skipping.")
        conn.close()
        return

    cursor.execute("SELECT product_id FROM products")
    product_ids = [row[0] for row in cursor.fetchall()]

    sales = []
    start_date = date.today() - timedelta(days=60)

    for day_offset in range(60):
        current_date = start_date + timedelta(days=day_offset)
        # 8 to 15 transactions per day
        for _ in range(random.randint(8, 15)):
            product_id = random.choice(product_ids)
            quantity = random.randint(1, 5)
            sales.append((product_id, quantity, current_date.isoformat()))

    cursor.executemany(
        "INSERT INTO sales (product_id, quantity, sale_date) VALUES (?,?,?)",
        sales
    )

    conn.commit()
    conn.close()
    print(f"✅ Seeded {len(sales)} sample sales transactions.")


if __name__ == "__main__":
    initialize_database()
    seed_pakistani_products()
    seed_sample_sales()