"""
Returns and Cancellation Tool.
Enables LLM to process returns, exchanges, and cancellations.
"""

import logging
from typing import Any, Dict
from datetime import datetime, timedelta

from app.tools.registry import Tool, ToolRegistry

logger = logging.getLogger(__name__)


async def initiate_return_handler(
    args: Dict[str, Any],
    session: Any = None
) -> Dict[str, Any]:
    """
    Start a return or exchange process.
    
    Args:
        args: {
            "order_id": str,    # Order to return
            "reason": str,      # Return reason
            "action": str       # refund, exchange, or store_credit
        }
    """
    from app.db.repositories.orders import OrderRepository
    from app.db.repositories.returns import ReturnRepository
    
    order_id = args.get("order_id")
    reason = args.get("reason")
    action = args.get("action", "refund")
    
    if not order_id and session and session.context.last_order_id:
        order_id = session.context.last_order_id
    
    if not order_id:
        return {"error": "Please provide an order ID for the return."}
    
    if not reason:
        return {"error": "Please provide a reason for the return."}
    
    try:
        order_repo = OrderRepository()
        return_repo = ReturnRepository()
        
        order = await order_repo.get_by_id(order_id)
        
        if not order:
            return {"error": f"Order '{order_id}' not found."}
        
        # Check if order is eligible for return
        if order.status == "cancelled":
            return {
                "order_id": order_id,
                "eligible": False,
                "message": "This order has already been cancelled."
            }
        
        if order.status == "returned":
            return {
                "order_id": order_id,
                "eligible": False,
                "message": "This order has already been returned."
            }
        
        # Check return window (e.g., 30 days)
        if order.created_at:
            days_since_order = (datetime.now() - order.created_at).days
            if days_since_order > 30:
                return {
                    "order_id": order_id,
                    "eligible": False,
                    "message": "This order is outside the 30-day return window."
                }
        
        # Create return request
        return_id = f"RET-{order_id[-6:]}-{datetime.now().strftime('%H%M%S')}"
        
        await return_repo.create({
            "id": return_id,
            "order_id": order_id,
            "reason": reason,
            "action_requested": action,
            "status": "pending",
            "refund_amount": order.total_amount if action == "refund" else None,
            "created_at": datetime.now()
        })
        
        # Update session
        if session:
            session.context.pending_action = f"return_{return_id}"
            session.context.awaiting_confirmation = True
        
        # Generate instructions based on order status
        if order.status in ["pending", "confirmed", "processing"]:
            instructions = "Your order hasn't shipped yet. We'll cancel it and process your refund."
        elif order.status in ["shipped", "out_for_delivery"]:
            instructions = "Please refuse delivery when your order arrives, or return it within 7 days of delivery."
        else:
            instructions = "Please pack the item securely and drop it at your nearest courier partner location. We'll email you a shipping label."
        
        return {
            "return_id": return_id,
            "order_id": order_id,
            "status": "initiated",
            "action": action,
            "refund_amount": order.total_amount,
            "instructions": instructions,
            "message": f"Return request {return_id} has been initiated. {instructions}"
        }
        
    except Exception as e:
        logger.error(f"Initiate return error: {e}")
        return {"error": str(e)}


async def cancel_order_handler(
    args: Dict[str, Any],
    session: Any = None
) -> Dict[str, Any]:
    """
    Cancel an order that hasn't shipped.
    
    Args:
        args: {
            "order_id": str,    # Order to cancel
            "reason": str       # Cancellation reason
        }
    """
    from app.db.repositories.orders import OrderRepository
    
    order_id = args.get("order_id")
    reason = args.get("reason", "Customer requested cancellation")
    
    if not order_id and session and session.context.last_order_id:
        order_id = session.context.last_order_id
    
    if not order_id:
        return {"error": "Please provide an order ID to cancel."}
    
    try:
        repo = OrderRepository()
        order = await repo.get_by_id(order_id)
        
        if not order:
            return {"error": f"Order '{order_id}' not found."}
        
        # Check if order can be cancelled
        if order.status in ["shipped", "out_for_delivery", "delivered"]:
            return {
                "order_id": order_id,
                "cancellable": False,
                "message": f"This order has already been {order.status}. Please request a return instead."
            }
        
        if order.status == "cancelled":
            return {
                "order_id": order_id,
                "cancellable": False,
                "message": "This order has already been cancelled."
            }
        
        # Cancel the order
        await repo.update(order_id, {
            "status": "cancelled",
            "updated_at": datetime.now()
        })
        
        return {
            "order_id": order_id,
            "status": "cancelled",
            "refund_amount": order.total_amount,
            "message": f"Order {order_id} has been cancelled. Your refund of ₹{order.total_amount:.2f} will be processed within 5-7 business days."
        }
        
    except Exception as e:
        logger.error(f"Cancel order error: {e}")
        return {"error": str(e)}


async def check_return_status_handler(
    args: Dict[str, Any],
    session: Any = None
) -> Dict[str, Any]:
    """
    Check status of a return request.
    
    Args:
        args: {
            "return_id": str    # Return ID to check
        }
    """
    from app.db.repositories.returns import ReturnRepository
    
    return_id = args.get("return_id")
    
    if not return_id:
        return {"error": "Please provide a return ID."}
    
    try:
        repo = ReturnRepository()
        return_request = await repo.get_by_id(return_id)
        
        if not return_request:
            return {"error": f"Return request '{return_id}' not found."}
        
        status_messages = {
            "pending": "Your return is being reviewed.",
            "approved": "Your return has been approved. Please ship the item back.",
            "item_received": "We've received your item and are processing your refund.",
            "refunded": "Your refund has been processed.",
            "rejected": "Your return request was not approved."
        }
        
        return {
            "return_id": return_id,
            "order_id": return_request.order_id,
            "status": return_request.status,
            "status_message": status_messages.get(return_request.status, "Status unknown"),
            "refund_amount": return_request.refund_amount,
            "created_at": return_request.created_at.strftime("%Y-%m-%d") if return_request.created_at else None
        }
        
    except Exception as e:
        logger.error(f"Check return status error: {e}")
        return {"error": str(e)}


async def get_return_eligibility_handler(
    args: Dict[str, Any],
    session: Any = None
) -> Dict[str, Any]:
    """
    Check if an order is eligible for return.
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
        
        # Check eligibility
        reasons = []
        eligible = True
        
        if order.status in ["cancelled", "returned"]:
            eligible = False
            reasons.append(f"Order is already {order.status}")
        
        if order.status == "pending":
            reasons.append("Order can be cancelled instead (hasn't shipped)")
        
        if order.created_at:
            days = (datetime.now() - order.created_at).days
            if days > 30:
                eligible = False
                reasons.append("Outside 30-day return window")
            else:
                reasons.append(f"{30 - days} days left in return window")
        
        return {
            "order_id": order_id,
            "eligible": eligible,
            "order_status": order.status,
            "details": reasons,
            "message": "This order is eligible for return." if eligible else "This order is not eligible for return."
        }
        
    except Exception as e:
        logger.error(f"Return eligibility error: {e}")
        return {"error": str(e)}


async def register_return_tools(registry: ToolRegistry):
    """Register all return-related tools."""
    
    # Initiate return
    registry.register(Tool(
        name="initiate_return",
        description="Start a return or exchange process for an order. Use when user wants to return an item, get a refund, or exchange.",
        parameters={
            "order_id": {
                "type": "string",
                "description": "Order ID for the return"
            },
            "reason": {
                "type": "string",
                "description": "Reason for the return",
                "enum": ["defective", "wrong_item", "not_as_described", "changed_mind", "size_issue", "quality_issue"]
            },
            "action": {
                "type": "string",
                "description": "What the customer wants",
                "enum": ["refund", "exchange", "store_credit"]
            }
        },
        handler=initiate_return_handler,
        required_params=["order_id", "reason"]
    ))
    
    # Cancel order
    registry.register(Tool(
        name="cancel_order",
        description="Cancel an order that hasn't shipped yet. Use when user wants to cancel their order.",
        parameters={
            "order_id": {
                "type": "string",
                "description": "Order ID to cancel"
            },
            "reason": {
                "type": "string",
                "description": "Reason for cancellation"
            }
        },
        handler=cancel_order_handler,
        required_params=["order_id"]
    ))
    
    # Check return status
    registry.register(Tool(
        name="check_return_status",
        description="Check the status of a return request. Use when user asks about their return status.",
        parameters={
            "return_id": {
                "type": "string",
                "description": "Return request ID to check"
            }
        },
        handler=check_return_status_handler,
        required_params=["return_id"]
    ))
    
    # Check return eligibility
    registry.register(Tool(
        name="check_return_eligibility",
        description="Check if an order is eligible for return. Use before initiating a return to verify eligibility.",
        parameters={
            "order_id": {
                "type": "string",
                "description": "Order ID to check eligibility for"
            }
        },
        handler=get_return_eligibility_handler,
        required_params=["order_id"]
    ))
    
    logger.info("Return tools registered")
