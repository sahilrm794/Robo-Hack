"""
Order Repository.
Data access layer for order operations.
"""

import logging
from typing import List, Optional
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.db.database import get_db
from app.db.models import Order, OrderItem

logger = logging.getLogger(__name__)


class OrderRepository:
    """Repository for order data operations."""
    
    async def get_by_id(self, order_id: str) -> Optional[Order]:
        """Get an order by ID with items."""
        async with get_db() as db:
            stmt = (
                select(Order)
                .options(selectinload(Order.items).selectinload(OrderItem.product))
                .where(Order.id == order_id)
            )
            result = await db.execute(stmt)
            return result.scalar_one_or_none()
    
    async def get_by_phone(
        self,
        phone: str,
        limit: int = 10
    ) -> List[Order]:
        """Get orders for a phone number."""
        async with get_db() as db:
            stmt = (
                select(Order)
                .options(selectinload(Order.items))
                .where(Order.user_phone == phone)
                .order_by(Order.created_at.desc())
                .limit(limit)
            )
            result = await db.execute(stmt)
            return list(result.scalars().all())
    
    async def get_by_status(
        self,
        status: str,
        limit: int = 50
    ) -> List[Order]:
        """Get orders by status."""
        async with get_db() as db:
            stmt = (
                select(Order)
                .where(Order.status == status)
                .order_by(Order.created_at.desc())
                .limit(limit)
            )
            result = await db.execute(stmt)
            return list(result.scalars().all())
    
    async def create(self, data: dict) -> Order:
        """Create a new order."""
        async with get_db() as db:
            # Extract items if present
            items_data = data.pop("items", [])
            
            order = Order(**data)
            db.add(order)
            await db.flush()
            
            # Create order items
            for item_data in items_data:
                item_data["order_id"] = order.id
                item = OrderItem(**item_data)
                db.add(item)
            
            await db.flush()
            return order
    
    async def update(self, order_id: str, data: dict) -> Optional[Order]:
        """Update an order."""
        async with get_db() as db:
            stmt = select(Order).where(Order.id == order_id)
            result = await db.execute(stmt)
            order = result.scalar_one_or_none()
            
            if order:
                for key, value in data.items():
                    if hasattr(order, key):
                        setattr(order, key, value)
                order.updated_at = datetime.utcnow()
                await db.flush()
            
            return order
    
    async def cancel(self, order_id: str) -> Optional[Order]:
        """Cancel an order."""
        return await self.update(order_id, {"status": "cancelled"})
    
    async def get_recent(self, limit: int = 20) -> List[Order]:
        """Get recent orders."""
        async with get_db() as db:
            stmt = (
                select(Order)
                .order_by(Order.created_at.desc())
                .limit(limit)
            )
            result = await db.execute(stmt)
            return list(result.scalars().all())
