"""
CRUD endpoints for Products and Purchase Orders.
"""

import logging
from datetime import date
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query

from database.db_manager import get_connection
from api.schemas.products import (
    ProductCreate, ProductUpdate, ProductResponse,
    StockUpdate,
)
from api.schemas.common import BaseResponse

log    = logging.getLogger(__name__)
router = APIRouter(prefix="/products", tags=["Products"])


@router.get("/", response_model=BaseResponse[List[ProductResponse]])
async def list_products(
    category  : Optional[str]  = Query(default=None),
    search    : Optional[str]  = Query(default=None),
    low_stock : bool            = Query(default=False),
    active_only: bool           = Query(default=True),
):
    """
    List products with optional filters.
    ?category=Tea & Beverages
    ?search=tapal
    ?low_stock=true  → only products below their threshold
    """
    try:
        conn   = get_connection()
        cursor = conn.cursor()

        query  = """
            SELECT *, ROUND((selling_price - cost_price) * 100.0 / selling_price, 2)
                AS margin_pct
            FROM products WHERE 1=1
        """
        params = []

        if active_only:
            query += " AND is_active = 1"
        if category:
            query += " AND LOWER(category) = LOWER(?)"
            params.append(category)
        if search:
            query += " AND LOWER(product_name) LIKE LOWER(?)"
            params.append(f"%{search}%")
        if low_stock:
            query += " AND stock <= low_stock_threshold"

        query += " ORDER BY category, product_name"
        cursor.execute(query, params)
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()

        return BaseResponse(success=True, data=rows)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{product_id}", response_model=BaseResponse[ProductResponse])
async def get_product(product_id: int):
    """Get single product by ID with full details."""
    try:
        conn   = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT *, ROUND((selling_price - cost_price)*100.0/selling_price,2)
                AS margin_pct
            FROM products WHERE product_id = ?
        """, (product_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Product not found")

        return BaseResponse(success=True, data=dict(row))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=BaseResponse[ProductResponse], status_code=201)
async def create_product(body: ProductCreate):
    """
    Add a new product.
    created_at is set automatically by server — client never sends it.
    """
    try:
        conn   = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO products
            (product_name, category, cost_price, selling_price,
             stock, supplier, low_stock_threshold, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            body.product_name, body.category,
            body.cost_price, body.selling_price,
            body.stock, body.supplier,
            body.low_stock_threshold, date.today().isoformat(),
        ))
        conn.commit()
        new_id = cursor.lastrowid

        cursor.execute("SELECT * FROM products WHERE product_id = ?", (new_id,))
        row = dict(cursor.fetchone())
        conn.close()

        log.info("Product created: %s (id=%d)", body.product_name, new_id)
        return BaseResponse(success=True, data=row,
                            message=f"Product '{body.product_name}' added successfully.")
    except Exception as e:
        if "UNIQUE constraint failed" in str(e):
            raise HTTPException(status_code=409,
                                detail=f"Product '{body.product_name}' already exists.")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{product_id}", response_model=BaseResponse[ProductResponse])
async def update_product(product_id: int, body: ProductUpdate):
    """Update product fields. Only sends fields you want to change."""
    try:
        conn   = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM products WHERE product_id = ?", (product_id,))
        existing = cursor.fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="Product not found")

        # Build dynamic UPDATE — only update fields that were sent
        updates = {}
        if body.product_name   is not None: updates["product_name"]        = body.product_name
        if body.category       is not None: updates["category"]             = body.category
        if body.cost_price     is not None: updates["cost_price"]           = body.cost_price
        if body.selling_price  is not None: updates["selling_price"]        = body.selling_price
        if body.supplier       is not None: updates["supplier"]             = body.supplier
        if body.low_stock_threshold is not None:
            updates["low_stock_threshold"] = body.low_stock_threshold

        if not updates:
            return BaseResponse(success=True, data=dict(existing),
                                message="No changes made.")

        updates["updated_at"] = date.today().isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        cursor.execute(
            f"UPDATE products SET {set_clause} WHERE product_id = ?",
            list(updates.values()) + [product_id]
        )
        conn.commit()

        cursor.execute("SELECT * FROM products WHERE product_id = ?", (product_id,))
        row = dict(cursor.fetchone())
        conn.close()

        return BaseResponse(success=True, data=row, message="Product updated.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{product_id}/stock", response_model=BaseResponse[ProductResponse])
async def update_stock(product_id: int, body: StockUpdate):
    """
    Update stock only.
    operation=add: new_stock = current + quantity (use negative to reduce)
    operation=set: new_stock = quantity (direct override)
    """
    try:
        conn   = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT stock FROM products WHERE product_id = ? AND is_active = 1",
                       (product_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Product not found")

        current_stock = row["stock"]
        if body.operation == "add":
            new_stock = max(0, current_stock + body.quantity)
        else:
            new_stock = max(0, body.quantity)

        cursor.execute(
            "UPDATE products SET stock = ?, updated_at = ? WHERE product_id = ?",
            (new_stock, date.today().isoformat(), product_id)
        )
        conn.commit()

        cursor.execute("SELECT * FROM products WHERE product_id = ?", (product_id,))
        updated = dict(cursor.fetchone())
        conn.close()

        return BaseResponse(
            success=True, data=updated,
            message=f"Stock updated: {current_stock} → {new_stock}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{product_id}", response_model=BaseResponse)
async def delete_product(product_id: int):
    """
    Soft delete — sets is_active=0.
    Product remains in DB for historical sales integrity.
    Cannot hard delete if sales records exist.
    """
    try:
        conn   = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT product_name FROM products WHERE product_id = ?",
                       (product_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Product not found")

        cursor.execute(
            "UPDATE products SET is_active = 0, updated_at = ? WHERE product_id = ?",
            (date.today().isoformat(), product_id)
        )
        conn.commit()
        conn.close()

        log.info("Product soft-deleted: id=%d", product_id)
        return BaseResponse(
            success=True,
            message=f"Product '{row['product_name']}' deactivated successfully."
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))