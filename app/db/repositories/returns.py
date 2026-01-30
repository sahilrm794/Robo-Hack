"""
Return Repository.
Data access layer for return request operations.
"""

import logging
from typing import List, Optional
from datetime import datetime

from sqlalchemy import select

from app.db.database import get_db
from app.db.models import Return

logger = logging.getLogger(__name__)


class ReturnRepository:
    """Repository for return data operations."""
    
    async def get_by_id(self, return_id: str) -> Optional[Return]:
        """Get a return request by ID."""
        async with get_db() as db:
            result = await db.execute(
                select(Return).where(Return.id == return_id)
            )
            return result.scalar_one_or_none()
    
    async def get_by_order(self, order_id: str) -> List[Return]:
        """Get return requests for an order."""
        async with get_db() as db:
            stmt = select(Return).where(Return.order_id == order_id)
            result = await db.execute(stmt)
            return list(result.scalars().all())
    
    async def get_by_status(
        self,
        status: str,
        limit: int = 50
    ) -> List[Return]:
        """Get return requests by status."""
        async with get_db() as db:
            stmt = (
                select(Return)
                .where(Return.status == status)
                .order_by(Return.created_at.desc())
                .limit(limit)
            )
            result = await db.execute(stmt)
            return list(result.scalars().all())
    
    async def create(self, data: dict) -> Return:
        """Create a new return request."""
        async with get_db() as db:
            return_request = Return(**data)
            db.add(return_request)
            await db.flush()
            return return_request
    
    async def update(self, return_id: str, data: dict) -> Optional[Return]:
        """Update a return request."""
        async with get_db() as db:
            stmt = select(Return).where(Return.id == return_id)
            result = await db.execute(stmt)
            return_request = result.scalar_one_or_none()
            
            if return_request:
                for key, value in data.items():
                    if hasattr(return_request, key):
                        setattr(return_request, key, value)
                await db.flush()
            
            return return_request
    
    async def approve(self, return_id: str, refund_amount: float) -> Optional[Return]:
        """Approve a return request."""
        return await self.update(return_id, {
            "status": "approved",
            "refund_amount": refund_amount,
            "processed_at": datetime.utcnow()
        })
    
    async def reject(self, return_id: str) -> Optional[Return]:
        """Reject a return request."""
        return await self.update(return_id, {
            "status": "rejected",
            "processed_at": datetime.utcnow()
        })
    
    async def complete(self, return_id: str) -> Optional[Return]:
        """Mark a return as completed."""
        return await self.update(return_id, {
            "status": "completed",
            "processed_at": datetime.utcnow()
        })
