"""
Purchase Orders CRUD endpoints.
"""

import logging
from datetime import date
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query

from database.db_manager import get_connection
from api.schemas.products import PurchaseOrderCreate, PurchaseOrderResponse
from api.schemas.common import BaseResponse

log    = logging.getLogger(__name__)
router = APIRouter(prefix="/orders", tags=["Purchase Orders"])


@router.get("/", response_model=BaseResponse[List[PurchaseOrderResponse]])
async def list_orders(
    status: Optional[str] = Query(default=None,
                                   description="pending / received / cancelled")
):
    """List purchase orders with optional status filter."""
    try:
        conn   = get_connection()
        cursor = conn.cursor()

        if status:
            cursor.execute("""
                SELECT po.*, p.product_name
                FROM purchase_orders po
                JOIN products p ON po.product_id = p.product_id
                WHERE po.status = ?
                ORDER BY po.order_date DESC
            """, (status,))
        else:
            cursor.execute("""
                SELECT po.*, p.product_name
                FROM purchase_orders po
                JOIN products p ON po.product_id = p.product_id
                ORDER BY po.order_date DESC
            """)

        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return BaseResponse(success=True, data=rows)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=BaseResponse[PurchaseOrderResponse], status_code=201)
async def create_order(body: PurchaseOrderCreate):
    """Create a new purchase order."""
    try:
        conn   = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT product_name FROM products WHERE product_id = ? AND is_active=1",
                       (body.product_id,))
        product = cursor.fetchone()
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")

        total_cost = round(body.quantity_ordered * body.cost_per_unit, 2)

        cursor.execute("""
            INSERT INTO purchase_orders
            (product_id, quantity_ordered, cost_per_unit, total_cost, supplier, order_date)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            body.product_id, body.quantity_ordered,
            body.cost_per_unit, total_cost,
            body.supplier, date.today().isoformat()
        ))
        conn.commit()
        order_id = cursor.lastrowid

        cursor.execute("""
            SELECT po.*, p.product_name FROM purchase_orders po
            JOIN products p ON po.product_id = p.product_id
            WHERE po.id = ?
        """, (order_id,))
        row = dict(cursor.fetchone())
        conn.close()

        log.info("Purchase order created: id=%d for product_id=%d", order_id, body.product_id)
        return BaseResponse(success=True, data=row,
                            message="Purchase order created.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{order_id}/receive", response_model=BaseResponse[PurchaseOrderResponse])
async def receive_order(order_id: int):
    """
    Mark order as received.
    Automatically increases product stock by quantity_ordered.
    This is the key business workflow: order received → stock updated.
    """
    try:
        conn   = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM purchase_orders WHERE id = ?", (order_id,))
        order = cursor.fetchone()

        if not order:
            raise HTTPException(status_code=404, detail="Order not found")
        if order["status"] == "received":
            raise HTTPException(status_code=400, detail="Order already marked as received")
        if order["status"] == "cancelled":
            raise HTTPException(status_code=400, detail="Cannot receive a cancelled order")

        # Update order status
        cursor.execute("""
            UPDATE purchase_orders
            SET status = 'received', received_date = ?
            WHERE id = ?
        """, (date.today().isoformat(), order_id))

        # Auto-update product stock
        cursor.execute("""
            UPDATE products
            SET stock = stock + ?, updated_at = ?
            WHERE product_id = ?
        """, (order["quantity_ordered"], date.today().isoformat(), order["product_id"]))

        conn.commit()

        cursor.execute("""
            SELECT po.*, p.product_name FROM purchase_orders po
            JOIN products p ON po.product_id = p.product_id
            WHERE po.id = ?
        """, (order_id,))
        row = dict(cursor.fetchone())
        conn.close()

        log.info("Order %d received — stock increased by %d", order_id, order["quantity_ordered"])
        return BaseResponse(
            success=True, data=row,
            message=f"Order received. Stock increased by {order['quantity_ordered']} units."
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{order_id}", response_model=BaseResponse)
async def cancel_order(order_id: int):
    """Cancel a pending order."""
    try:
        conn   = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT status FROM purchase_orders WHERE id = ?", (order_id,))
        order = cursor.fetchone()

        if not order:
            raise HTTPException(status_code=404, detail="Order not found")
        if order["status"] != "pending":
            raise HTTPException(status_code=400,
                                detail=f"Cannot cancel order with status '{order['status']}'")

        cursor.execute(
            "UPDATE purchase_orders SET status = 'cancelled' WHERE id = ?",
            (order_id,)
        )
        conn.commit()
        conn.close()

        return BaseResponse(success=True, message=f"Order {order_id} cancelled.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))