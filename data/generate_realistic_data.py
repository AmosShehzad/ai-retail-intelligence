"""
data/generate_realistic_data.py

Generates realistic Pakistani kiryana store data:
- 60 products with real 2026 prices (Carrefour/Imtiaz verified)
- 365 days of sales (Jan 1 2024 - Dec 31 2024)
- Realistic patterns: Ramadan, summer, salary week, weekends
- sale_price stored at time of sale for accurate historical revenue
- 5 products at critically low stock for meaningful alerts
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
import math
from datetime import date, timedelta
from database.db_manager import get_connection, initialize_database

random.seed(42)

# ── 60 verified Pakistani products ────────────────────────────────────────────
# (name, category, cost, selling, initial_stock, supplier, low_stock_threshold)
PRODUCTS = [
    # Tea & Beverages
    ("Tapal Danedar 200g",         "Tea & Beverages",  185, 235, 85, "Tapal Foods",        15),
    ("Tapal Danedar 500g",         "Tea & Beverages",  430, 540, 40, "Tapal Foods",        10),
    ("Tapal Family Mixture 95g",   "Tea & Beverages",   88, 118, 60, "Tapal Foods",        12),
    ("Lipton Yellow Label 200g",   "Tea & Beverages",  195, 250, 50, "Unilever Pakistan",  10),
    ("Nescafe Classic 50g",        "Tea & Beverages",  320, 410, 30, "Nestle Pakistan",     8),
    ("Nestle Milo 400g",           "Tea & Beverages",  530, 670, 25, "Nestle Pakistan",     5),
    ("Tang Orange 500g",           "Tea & Beverages",  178, 230, 45, "Mondelez Pakistan",  10),
    ("Rooh Afza 800ml",            "Tea & Beverages",  345, 440, 20, "Hamdard",             5),

    # Spices & Masala
    ("Shan Biryani Masala 60g",    "Spices & Masala",   75,  98,110, "Shan Foods",         20),
    ("Shan Karahi Masala 50g",     "Spices & Masala",   68,  88, 95, "Shan Foods",         15),
    ("Shan Nihari Masala 60g",     "Spices & Masala",   70,  92, 80, "Shan Foods",         15),
    ("National Biryani Masala 55g","Spices & Masala",   70,  92, 70, "National Foods",     12),
    ("National Karahi Masala 50g", "Spices & Masala",   65,  85, 65, "National Foods",     12),
    ("Ahmed Chaat Masala 100g",    "Spices & Masala",   58,  78, 90, "Ahmed Foods",        15),
    ("Mehran Tikka Masala 50g",    "Spices & Masala",   60,  80, 75, "Mehran Foods",       12),

    # Dairy
    ("Nestle MilkPak 1L",          "Dairy",            172, 220, 60, "Nestle Pakistan",    15),
    ("Olpers Milk 1L",             "Dairy",            165, 215, 55, "Engro Foods",        15),
    ("Haleeb Milk 1L",             "Dairy",            158, 208, 45, "Haleeb Foods",       12),
    ("Nestle Cream 200ml",         "Dairy",            118, 158, 35, "Nestle Pakistan",     8),
    ("Olpers Cream 200ml",         "Dairy",            112, 152, 30, "Engro Foods",         8),
    ("Good Milk Yogurt 400g",      "Dairy",             88, 120, 40, "Good Milk",          10),

    # Cooking Oil
    ("Dalda Cooking Oil 1L",       "Cooking Oil",      395, 490, 45, "Dalda Foods",        10),
    ("Sufi Cooking Oil 1L",        "Cooking Oil",      385, 478, 40, "Sufi Group",         10),
    ("Dalda Banaspati 1kg",        "Cooking Oil",      435, 530, 30, "Dalda Foods",         8),
    ("Habib Cooking Oil 1L",       "Cooking Oil",      378, 470, 35, "Habib Oil",           8),
    ("Seasons Canola Oil 1L",      "Cooking Oil",      415, 510, 25, "Seasons",             5),

    # Detergents
    ("Surf Excel 500g",            "Detergents",       310, 395, 50, "Unilever Pakistan",  10),
    ("Surf Excel 1kg",             "Detergents",       590, 740, 30, "Unilever Pakistan",   8),
    ("Ariel 500g",                 "Detergents",       295, 378, 35, "P&G Pakistan",        8),
    ("Bonus Washing Powder 1kg",   "Detergents",       225, 298, 60, "Colgate-Palmolive",  12),
    ("Brite Detergent 500g",       "Detergents",       175, 238, 55, "Colgate-Palmolive",  12),

    # Personal Care
    ("Lux Soap 150g",              "Personal Care",     85, 118,120, "Unilever Pakistan",  20),
    ("Safeguard Soap 175g",        "Personal Care",     95, 132,100, "P&G Pakistan",       20),
    ("Dettol Soap 175g",           "Personal Care",     98, 138, 90, "Reckitt Pakistan",   15),
    ("Lifebuoy Soap 135g",         "Personal Care",     75, 108, 85, "Unilever Pakistan",  15),
    ("Dettol Antiseptic 60ml",     "Personal Care",    128, 175, 65, "Reckitt Pakistan",   10),
    ("Colgate Total 75ml",         "Personal Care",    112, 155, 80, "Colgate-Palmolive",  15),
    ("Close Up 80ml",              "Personal Care",     95, 132, 75, "Unilever Pakistan",  15),
    ("Sunsilk Shampoo 180ml",      "Personal Care",    198, 265, 40, "Unilever Pakistan",   8),

    # Instant Food
    ("Knorr Chicken Noodles 66g",  "Instant Food",      55,  80,180, "Unilever Pakistan",  30),
    ("Knorr Masala Noodles 66g",   "Instant Food",      55,  80,160, "Unilever Pakistan",  30),
    ("Indomie Noodles 70g",        "Instant Food",      48,  70,140, "Kolson",             25),
    ("Shan Soup Mix 60g",          "Instant Food",      68,  92, 90, "Shan Foods",         15),
    ("National Ketchup 300g",      "Instant Food",     118, 158, 70, "National Foods",     12),
    ("Knorr Ketchup 300g",         "Instant Food",     122, 165, 65, "Unilever Pakistan",  12),

    # Snacks
    ("Peek Freans Sooper 112g",    "Snacks",            68,  95,150, "EBM",                25),
    ("Peek Freans Rio 90g",        "Snacks",            58,  82,120, "EBM",                20),
    ("Bisconni Choc Chip 82g",     "Snacks",            62,  88,100, "Bisconni",           20),
    ("Kolson Slanty 30g",          "Snacks",            42,  65,200, "Kolson",             35),
    ("Lays Classic 34g",           "Snacks",            40,  62,220, "PepsiCo Pakistan",   35),
    ("English Biscuits 100g",      "Snacks",            55,  78,130, "EBM",                20),
    ("Candyland Cocomo 40g",       "Snacks",            35,  55,180, "Candyland",          30),

    # Beverages
    ("Pepsi 1.5L",                 "Beverages",        115, 155, 60, "PepsiCo Pakistan",   15),
    ("Coca Cola 1.5L",             "Beverages",        118, 158, 55, "Coca-Cola Pakistan", 15),
    ("7UP 1.5L",                   "Beverages",        112, 152, 50, "PepsiCo Pakistan",   12),
    ("Sprite 345ml Can",           "Beverages",         82, 115, 80, "Coca-Cola Pakistan", 15),
    ("Pakola Ice Cream Soda 250ml","Beverages",         48,  70,100, "Mehran Bottlers",    20),

    # Condiments
    ("National Mango Pickle 400g", "Condiments",       162, 215, 40, "National Foods",      8),
    ("Shan Mixed Pickle 400g",     "Condiments",       158, 210, 38, "Shan Foods",          8),
    ("National Chilli Sauce 300g", "Condiments",       122, 165, 45, "National Foods",     10),
]


def get_seasonal_multiplier(sale_date: date, product_name: str, category: str) -> float:
    """
    Returns a sales volume multiplier based on:
    - Month (Ramadan, summer, eid)
    - Day of week (Friday-Saturday peak)
    - Day of month (salary week boost)
    - Product type (beverages spike in summer)
    """
    multiplier = 1.0
    month = sale_date.month
    day   = sale_date.day
    weekday = sale_date.weekday()  # 0=Monday, 4=Friday, 5=Saturday

    # Ramadan effect (approximately March-April 2024)
    if month in (3, 4):
        if category in ("Tea & Beverages", "Spices & Masala", "Instant Food"):
            multiplier *= 1.8
        if "Rooh Afza" in product_name:
            multiplier *= 3.5  # Rooh Afza is THE Ramadan drink
        if "Shan" in product_name or "National" in product_name:
            multiplier *= 2.2  # masala packets spike massively

    # Eid ul Fitr shopping (end of Ramadan ~April 10)
    if month == 4 and 5 <= day <= 12:
        multiplier *= 1.5

    # Summer beverages (June-August)
    if month in (6, 7, 8):
        if category == "Beverages":
            multiplier *= 2.5
        if "Pepsi" in product_name or "Coca Cola" in product_name or "7UP" in product_name:
            multiplier *= 1.8
        if category == "Tea & Beverages" and "Tang" in product_name:
            multiplier *= 2.0

    # Winter tea (November-February)
    if month in (11, 12, 1, 2):
        if "Tapal" in product_name or "Lipton" in product_name:
            multiplier *= 1.4

    # Friday-Saturday shopping peak
    if weekday in (4, 5):
        multiplier *= 1.4

    # Salary week (1st-5th of month)
    if 1 <= day <= 5:
        multiplier *= 1.3

    # End of month slowdown (25th-31st)
    if day >= 25:
        multiplier *= 0.8

    return multiplier


# Per-product base daily sales velocity (units/day on an average day)
PRODUCT_VELOCITY = {
    # Fast movers — staples
    "Tapal Danedar 200g"        : 4.5,
    "Knorr Chicken Noodles 66g" : 5.0,
    "Knorr Masala Noodles 66g"  : 4.5,
    "Lays Classic 34g"          : 6.0,
    "Kolson Slanty 30g"         : 5.5,
    "Lux Soap 150g"             : 4.0,
    "Safeguard Soap 175g"       : 3.5,
    "Peek Freans Sooper 112g"   : 4.0,
    "Shan Biryani Masala 60g"   : 3.5,
    "Nestle MilkPak 1L"         : 5.0,
    "Olpers Milk 1L"            : 4.5,
    "Pepsi 1.5L"                : 3.5,
    "Coca Cola 1.5L"            : 3.0,
    # Medium movers
    "Surf Excel 500g"           : 2.5,
    "Tapal Family Mixture 95g"  : 2.5,
    "National Biryani Masala 55g":2.0,
    "Indomie Noodles 70g"       : 3.0,
    "Candyland Cocomo 40g"      : 4.0,
    "Pakola Ice Cream Soda 250ml":3.0,
    "Lifebuoy Soap 135g"        : 2.5,
    "Peek Freans Rio 90g"       : 2.8,
    # Slow movers — premium/specialty
    "Nestle Milo 400g"          : 0.8,
    "Nescafe Classic 50g"       : 0.9,
    "Surf Excel 1kg"            : 1.0,
    "Seasons Canola Oil 1L"     : 0.7,
    "Sunsilk Shampoo 180ml"     : 0.9,
    "Rooh Afza 800ml"           : 0.6,
    "Dalda Banaspati 1kg"       : 0.8,
}

DEFAULT_VELOCITY = 1.5  # for products not in above dict


def generate_sales_for_year(products: list) -> list:
    """
    Generates realistic daily sales transactions for all 365 days of 2024.
    Returns list of (product_id, quantity, sale_price, sale_date) tuples.
    """
    start_date = date(2024, 1, 1)
    all_sales   = []

    for product in products:
        pid          = product["product_id"]
        name         = product["product_name"]
        category     = product["category"]
        base_price   = product["selling_price"]
        base_velocity= PRODUCT_VELOCITY.get(name, DEFAULT_VELOCITY)

        for day_offset in range(366):
            current_date = start_date + timedelta(days=day_offset)
            multiplier   = get_seasonal_multiplier(current_date, name, category)
            expected_qty = base_velocity * multiplier

            # Add randomness: ±30% around expected
            actual_qty = random.gauss(expected_qty, expected_qty * 0.3)
            actual_qty = max(0, round(actual_qty))

            if actual_qty == 0:
                continue

            # Slight price variation (±2%) to simulate real-world pricing
            sale_price = round(base_price * random.uniform(0.98, 1.02), 2)

            # Split large quantities into 1-3 transactions (realistic)
            if actual_qty > 6:
                splits = random.randint(2, 3)
                for _ in range(splits):
                    qty = max(1, round(actual_qty / splits))
                    all_sales.append((pid, qty, sale_price, current_date.isoformat()))
            else:
                all_sales.append((pid, actual_qty, sale_price, current_date.isoformat()))

    return all_sales


def set_realistic_stock_levels(conn):
    """
    After generating sales, sets realistic current stock levels.
    Intentionally sets 5 products at critically low stock
    to make dashboard alerts meaningful.
    """
    cursor = conn.cursor()

    # Critically low — will trigger CRITICAL alerts
    critical_low = [
        ("Tapal Danedar 200g",     4),   # best seller, always running out
        ("Knorr Chicken Noodles 66g", 6),
        ("Pepsi 1.5L",             5),
        ("Lux Soap 150g",          3),
        ("Nestle MilkPak 1L",      7),
    ]
    for name, stock in critical_low:
        cursor.execute(
            "UPDATE products SET stock=? WHERE product_name=?",
            (stock, name)
        )

    # Moderate stock
    moderate = [
        ("Surf Excel 500g", 18), ("Shan Biryani Masala 60g", 22),
        ("Colgate Total 75ml", 15), ("Olpers Milk 1L", 19),
    ]
    for name, stock in moderate:
        cursor.execute(
            "UPDATE products SET stock=? WHERE product_name=?",
            (stock, name)
        )

    conn.commit()


def run_generation():
    print("=" * 55)
    print("REALISTIC DATA GENERATION STARTED")
    print("=" * 55)

    initialize_database()
    conn   = get_connection()
    cursor = conn.cursor()

    # Step 1: Insert products
    print("Step 1/3: Inserting 60 Pakistani products...")
    cursor.executemany("""
        INSERT OR IGNORE INTO products
        (product_name, category, cost_price, selling_price,
         stock, supplier, low_stock_threshold)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, PRODUCTS)
    conn.commit()

    # Get products back with their IDs
    cursor.execute("SELECT product_id, product_name, category, selling_price FROM products")
    product_rows = cursor.fetchall()
    products_list = [
        {
            "product_id"  : r["product_id"],
            "product_name": r["product_name"],
            "category"    : r["category"],
            "selling_price": r["selling_price"],
        }
        for r in product_rows
    ]
    print(f"  → {len(products_list)} products inserted.")

    # Step 2: Generate 365 days of sales
    print("Step 2/3: Generating 365 days of realistic sales...")
    sales = generate_sales_for_year(products_list)
    cursor.executemany(
        "INSERT INTO sales (product_id, quantity, sale_price, sale_date) VALUES (?,?,?,?)",
        sales
    )
    conn.commit()
    print(f"  → {len(sales):,} sales transactions generated.")

    # Step 3: Set realistic stock levels
    print("Step 3/3: Setting realistic stock levels with alerts...")
    set_realistic_stock_levels(conn)
    conn.close()

    print("-" * 55)
    print("GENERATION COMPLETE ✅")
    print(f"  Products : 60")
    print(f"  Sales    : {len(sales):,}")
    print(f"  Period   : Jan 1 2024 — Dec 31 2024")
    print("=" * 55)


if __name__ == "__main__":
    run_generation()