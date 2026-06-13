import sqlite3

try:
    from .db_manager import get_connection
except ImportError:
    from db_manager import get_connection


def get_top_selling_products(limit=10):
    """Returns products ranked by total quantity sold."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            p.product_name,
            p.category,
            SUM(s.quantity) AS total_sold,
            ROUND(SUM(s.quantity * p.selling_price), 2) AS total_revenue
        FROM sales s
        JOIN products p ON s.product_id = p.product_id
        GROUP BY p.product_id
        ORDER BY total_sold DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_low_stock_products(threshold=20):
    """Returns products with stock below the given threshold."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT product_name, category, stock, supplier
        FROM products
        WHERE stock <= ?
        ORDER BY stock ASC
    """, (threshold,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_profit_margins():
    """Returns each product with its profit margin percentage."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            product_name,
            category,
            cost_price,
            selling_price,
            ROUND((selling_price - cost_price) * 100.0 / cost_price, 2) AS margin_pct
        FROM products
        ORDER BY margin_pct DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_daily_revenue(days=30):
    """Returns daily revenue for the last N days."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            s.sale_date,
            ROUND(SUM(s.quantity * p.selling_price), 2) AS daily_revenue
        FROM sales s
        JOIN products p ON s.product_id = p.product_id
        WHERE s.sale_date >= DATE('now', ? || ' days')
        GROUP BY s.sale_date
        ORDER BY s.sale_date ASC
    """, (f"-{days}",))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_competitor_prices(product_name=None):
    """Returns competitor prices, optionally filtered by product name."""
    conn = get_connection()
    cursor = conn.cursor()
    if product_name:
        cursor.execute("""
            SELECT product_name, store_name, price, scrape_date
            FROM competitor_prices
            WHERE LOWER(product_name) LIKE LOWER(?)
            ORDER BY price ASC
        """, (f"%{product_name}%",))
    else:
        cursor.execute("""
            SELECT product_name, store_name, price, scrape_date
            FROM competitor_prices
            ORDER BY scrape_date DESC
        """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def log_query(question: str, answer: str):
    """Saves an AI query and its answer to the audit log."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO query_logs (question, answer) VALUES (?, ?)",
        (question, answer)
    )
    conn.commit()
    conn.close()