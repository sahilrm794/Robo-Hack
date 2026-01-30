"""
Policy Repository.
Data access layer for policy operations.
"""

import logging
from typing import List, Optional

from sqlalchemy import select, or_

from app.db.database import get_db
from app.db.models import Policy

logger = logging.getLogger(__name__)


class PolicyRepository:
    """Repository for policy data operations."""
    
    async def get_by_id(self, policy_id: str) -> Optional[Policy]:
        """Get a policy by ID."""
        async with get_db() as db:
            result = await db.execute(
                select(Policy).where(Policy.id == policy_id)
            )
            return result.scalar_one_or_none()
    
    async def get_by_type(self, policy_type: str) -> Optional[Policy]:
        """Get a policy by type."""
        async with get_db() as db:
            result = await db.execute(
                select(Policy).where(Policy.type == policy_type)
            )
            return result.scalar_one_or_none()
    
    async def get_all(self) -> List[Policy]:
        """Get all policies."""
        async with get_db() as db:
            result = await db.execute(select(Policy))
            return list(result.scalars().all())
    
    async def search(
        self,
        query: str,
        limit: int = 5
    ) -> List[Policy]:
        """Search policies by content."""
        async with get_db() as db:
            search_term = f"%{query}%"
            
            stmt = select(Policy).where(
                or_(
                    Policy.title.ilike(search_term),
                    Policy.content.ilike(search_term),
                    Policy.title_hi.ilike(search_term),
                    Policy.content_hi.ilike(search_term)
                )
            ).limit(limit)
            
            result = await db.execute(stmt)
            return list(result.scalars().all())
    
    async def create(self, data: dict) -> Policy:
        """Create a new policy."""
        async with get_db() as db:
            policy = Policy(**data)
            db.add(policy)
            await db.flush()
            return policy
    
    async def update(self, policy_type: str, data: dict) -> Optional[Policy]:
        """Update a policy."""
        async with get_db() as db:
            policy = await self.get_by_type(policy_type)
            if policy:
                for key, value in data.items():
                    if hasattr(policy, key):
                        setattr(policy, key, value)
                await db.flush()
            return policy
