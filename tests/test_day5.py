import sys
from pathlib import Path
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from database.db_manager import get_connection


def test_invalid_margins():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*) FROM products
        WHERE cost_price >= selling_price
    """)
    count = cursor.fetchone()[0]
    conn.close()

    assert count == 0, f"Invalid margins found: {count}"


def test_negative_stock():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM products WHERE stock < 0")
    count = cursor.fetchone()[0]
    conn.close()

    assert count == 0, f"Negative stock found: {count}"


def test_orphan_sales():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*) FROM sales
        WHERE product_id NOT IN (SELECT product_id FROM products)
    """)
    count = cursor.fetchone()[0]
    conn.close()

    assert count == 0, f"Orphan sales found: {count}"


def test_duplicate_products():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT product_name, category, COUNT(*) c
        FROM products
        GROUP BY product_name, category
        HAVING c > 1
    """)
    rows = cursor.fetchall()
    conn.close()

    assert len(rows) == 0, f"Duplicate products found: {len(rows)}"
print("Everything ran perfectly")