import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from database.db_manager import get_connection

conn = get_connection()
cursor = conn.cursor()

# Should all return 0
cursor.execute("SELECT COUNT(*) FROM products WHERE cost_price >= selling_price")
print("Invalid margins:", cursor.fetchone()[0])

cursor.execute("SELECT COUNT(*) FROM products WHERE stock < 0")
print("Negative stock:", cursor.fetchone()[0])

cursor.execute("""
    SELECT COUNT(*) FROM sales
    WHERE product_id NOT IN (SELECT product_id FROM products)
""")
print("Orphan sales:", cursor.fetchone()[0])

cursor.execute("""
    SELECT product_name, category, COUNT(*) c
    FROM products GROUP BY product_name, category HAVING c > 1
""")
print("Duplicate products:", len(cursor.fetchall()))

conn.close()