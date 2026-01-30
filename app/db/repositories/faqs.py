"""
FAQ Repository.
Data access layer for FAQ operations.
"""

import logging
from typing import List, Optional

from sqlalchemy import select, or_

from app.db.database import get_db
from app.db.models import FAQ

logger = logging.getLogger(__name__)


class FAQRepository:
    """Repository for FAQ data operations."""
    
    async def get_by_id(self, faq_id: str) -> Optional[FAQ]:
        """Get an FAQ by ID."""
        async with get_db() as db:
            result = await db.execute(
                select(FAQ).where(FAQ.id == faq_id)
            )
            return result.scalar_one_or_none()
    
    async def get_for_product(
        self,
        product_id: str,
        topic: Optional[str] = None,
        limit: int = 10
    ) -> List[FAQ]:
        """Get FAQs for a specific product."""
        async with get_db() as db:
            stmt = select(FAQ).where(FAQ.product_id == product_id)
            
            if topic:
                stmt = stmt.where(FAQ.topic == topic)
            
            stmt = stmt.limit(limit)
            
            result = await db.execute(stmt)
            return list(result.scalars().all())
    
    async def get_general(
        self,
        topic: Optional[str] = None,
        limit: int = 10
    ) -> List[FAQ]:
        """Get general (non-product specific) FAQs."""
        async with get_db() as db:
            stmt = select(FAQ).where(FAQ.is_general == True)
            
            if topic:
                stmt = stmt.where(FAQ.topic == topic)
            
            stmt = stmt.limit(limit)
            
            result = await db.execute(stmt)
            return list(result.scalars().all())
    
    async def search(
        self,
        query: str,
        topic: Optional[str] = None,
        limit: int = 10
    ) -> List[FAQ]:
        """Search FAQs by question or answer content."""
        async with get_db() as db:
            search_term = f"%{query}%"
            
            stmt = select(FAQ).where(
                or_(
                    FAQ.question.ilike(search_term),
                    FAQ.answer.ilike(search_term),
                    FAQ.question_hi.ilike(search_term),
                    FAQ.question_bn.ilike(search_term),
                    FAQ.question_mr.ilike(search_term)
                )
            )
            
            if topic:
                stmt = stmt.where(FAQ.topic == topic)
            
            stmt = stmt.limit(limit)
            
            result = await db.execute(stmt)
            return list(result.scalars().all())
    
    async def get_all(self, limit: int = 100) -> List[FAQ]:
        """Get all FAQs."""
        async with get_db() as db:
            stmt = select(FAQ).limit(limit)
            result = await db.execute(stmt)
            return list(result.scalars().all())
    
    async def create(self, data: dict) -> FAQ:
        """Create a new FAQ."""
        async with get_db() as db:
            faq = FAQ(**data)
            db.add(faq)
            await db.flush()
            return faq
    
    async def bulk_create(self, faqs: List[dict]) -> List[FAQ]:
        """Create multiple FAQs."""
        async with get_db() as db:
            created = []
            for faq_data in faqs:
                faq = FAQ(**faq_data)
                db.add(faq)
                created.append(faq)
            await db.flush()
            return created
