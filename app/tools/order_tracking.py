"""
Order Tracking Tool.
Enables LLM to track and retrieve order information.
"""

import logging
from typing import Any, Dict
from datetime import datetime

from app.tools.registry import Tool, ToolRegistry

logger = logging.getLogger(__name__)


async def track_order_handler(
    args: Dict[str, Any],
    session: Any = None
) -> Dict[str, Any]:
    """
    Track an order by order ID.
    
    Args:
        args: {
            "order_id": str,        # Order ID to track
            "phone_number": str     # Optional phone for verification
        }
    """
    from app.db.repositories.orders import OrderRepository
    
    order_id = args.get("order_id")
    phone = args.get("phone_number")
    
    # Use last order from context if not provided
    if not order_id and session and session.context.last_order_id:
        order_id = session.context.last_order_id
    
    if not order_id:
        return {"error": "Please provide an order ID to track."}
    
    try:
        repo = OrderRepository()
        order = await repo.get_by_id(order_id)
        
        if not order:
            return {
                "order_id": order_id,
                "error": f"Order '{order_id}' not found. Please check the order ID."
            }
        
        # Verify phone if provided
        if phone and order.user_phone != phone:
            return {
                "order_id": order_id,
                "error": "Phone number doesn't match our records for this order."
            }
        
        # Update session context
        if session:
            session.context.last_order_id = order_id
        
        # Format order items
        items = []
        for item in order.items:
            items.append({
                "product_name": item.product.name if item.product else "Unknown Product",
                "quantity": item.quantity,
                "price": item.unit_price
            })
        
        # Calculate status message
        status_messages = {
            "pending": "Your order is being processed.",
            "confirmed": "Your order has been confirmed and will be shipped soon.",
            "processing": "Your order is being prepared for shipping.",
            "shipped": "Your order has been shipped and is on its way!",
            "out_for_delivery": "Your order is out for delivery today!",
            "delivered": "Your order has been delivered.",
            "cancelled": "This order has been cancelled.",
            "returned": "This order has been returned."
        }
        
        return {
            "order_id": order.id,
            "status": order.status,
            "status_message": status_messages.get(order.status, "Order status unknown."),
            "items": items,
            "total_amount": order.total_amount,
            "shipping_address": order.shipping_address,
            "estimated_delivery": order.estimated_delivery.strftime("%Y-%m-%d") if order.estimated_delivery else None,
            "tracking_number": order.tracking_number,
            "shipping_carrier": order.shipping_carrier,
            "created_at": order.created_at.strftime("%Y-%m-%d %H:%M") if order.created_at else None
        }
        
    except Exception as e:
        logger.error(f"Order tracking error: {e}")
        return {"error": str(e)}


async def get_order_history_handler(
    args: Dict[str, Any],
    session: Any = None
) -> Dict[str, Any]:
    """
    Get order history for a phone number.
    
    Args:
        args: {
            "phone_number": str,    # Phone number to look up
            "limit": int            # Optional limit on results
        }
    """
    from app.db.repositories.orders import OrderRepository
    
    phone = args.get("phone_number")
    limit = args.get("limit", 5)
    
    # Use phone from context if not provided
    if not phone and session and session.context.user_phone:
        phone = session.context.user_phone
    
    if not phone:
        return {"error": "Please provide a phone number to look up orders."}
    
    try:
        repo = OrderRepository()
        orders = await repo.get_by_phone(phone, limit=limit)
        
        if not orders:
            return {
                "phone_number": phone,
                "orders": [],
                "message": "No orders found for this phone number."
            }
        
        # Update context
        if session:
            session.context.user_phone = phone
        
        formatted_orders = []
        for order in orders:
            formatted_orders.append({
                "order_id": order.id,
                "status": order.status,
                "total_amount": order.total_amount,
                "item_count": len(order.items),
                "created_at": order.created_at.strftime("%Y-%m-%d") if order.created_at else None
            })
        
        return {
            "phone_number": phone,
            "orders": formatted_orders,
            "count": len(formatted_orders)
        }
        
    except Exception as e:
        logger.error(f"Order history error: {e}")
        return {"error": str(e)}


async def get_order_items_handler(
    args: Dict[str, Any],
    session: Any = None
) -> Dict[str, Any]:
    """
    Get items in a specific order.
    """
    from app.db.repositories.orders import OrderRepository
    
    order_id = args.get("order_id")
    
    if not order_id and session and session.context.last_order_id:
        order_id = session.context.last_order_id
    
    if not order_id:
        return {"error": "Please provide an order ID."}
    
    try:
        repo = OrderRepository()
        order = await repo.get_by_id(order_id)
        
        if not order:
            return {"error": f"Order '{order_id}' not found."}
        
        items = []
        for item in order.items:
            items.append({
                "product_id": item.product_id,
                "product_name": item.product.name if item.product else "Unknown",
                "quantity": item.quantity,
                "unit_price": item.unit_price,
                "subtotal": item.quantity * item.unit_price
            })
        
        return {
            "order_id": order_id,
            "items": items,
            "total_amount": order.total_amount
        }
        
    except Exception as e:
        logger.error(f"Get order items error: {e}")
        return {"error": str(e)}


async def register_order_tools(registry: ToolRegistry):
    """Register all order-related tools."""
    
    # Track order
    registry.register(Tool(
        name="track_order",
        description="Track an order's status and delivery information. Use when user asks about their order status, where their order is, or when it will arrive.",
        parameters={
            "order_id": {
                "type": "string",
                "description": "Order ID to track (e.g., ORD-12345)"
            },
            "phone_number": {
                "type": "string",
                "description": "Phone number for verification (optional)"
            }
        },
        handler=track_order_handler,
        required_params=["order_id"]
    ))
    
    # Get order history
    registry.register(Tool(
        name="get_order_history",
        description="Get list of orders for a phone number. Use when user asks about their past orders or order history.",
        parameters={
            "phone_number": {
                "type": "string",
                "description": "Phone number to look up orders for"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of orders to return",
                "default": 5
            }
        },
        handler=get_order_history_handler,
        required_params=["phone_number"]
    ))
    
    # Get order items
    registry.register(Tool(
        name="get_order_items",
        description="Get the items in a specific order. Use when user asks what was in their order.",
        parameters={
            "order_id": {
                "type": "string",
                "description": "Order ID to get items for"
            }
        },
        handler=get_order_items_handler,
        required_params=["order_id"]
    ))
    
    logger.info("Order tools registered")
