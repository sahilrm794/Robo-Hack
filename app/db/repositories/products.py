"""
Product Repository.
Data access layer for product operations.
"""

import logging
from typing import List, Optional

from sqlalchemy import select, or_
from sqlalchemy.orm import selectinload

from app.db.database import get_db
from app.db.models import Product

logger = logging.getLogger(__name__)


class ProductRepository:
    """Repository for product data operations."""
    
    async def get_by_id(self, product_id: str) -> Optional[Product]:
        """Get a product by ID."""
        async with get_db() as db:
            result = await db.execute(
                select(Product).where(Product.id == product_id)
            )
            return result.scalar_one_or_none()
    
    async def search(
        self,
        query: str,
        category: Optional[str] = None,
        price_min: Optional[float] = None,
        price_max: Optional[float] = None,
        limit: int = 10
    ) -> List[Product]:
        """
        Search products by name, description, or category.
        """
        async with get_db() as db:
            stmt = select(Product)
            
            # Text search (simple LIKE for SQLite)
            if query:
                search_term = f"%{query}%"
                stmt = stmt.where(
                    or_(
                        Product.name.ilike(search_term),
                        Product.description.ilike(search_term),
                        Product.name_hi.ilike(search_term),
                        Product.name_bn.ilike(search_term),
                        Product.name_mr.ilike(search_term)
                    )
                )
            
            # Category filter
            if category:
                stmt = stmt.where(Product.category == category)
            
            # Price filters
            if price_min is not None:
                stmt = stmt.where(Product.price >= price_min)
            if price_max is not None:
                stmt = stmt.where(Product.price <= price_max)
            
            # Limit results
            stmt = stmt.limit(limit)
            
            result = await db.execute(stmt)
            return list(result.scalars().all())
    
    async def get_by_category(
        self,
        category: str,
        limit: int = 10
    ) -> List[Product]:
        """Get products in a category."""
        async with get_db() as db:
            stmt = (
                select(Product)
                .where(Product.category == category)
                .limit(limit)
            )
            result = await db.execute(stmt)
            return list(result.scalars().all())
    
    async def get_all(self, limit: int = 100) -> List[Product]:
        """Get all products."""
        async with get_db() as db:
            stmt = select(Product).limit(limit)
            result = await db.execute(stmt)
            return list(result.scalars().all())
    
    async def create(self, data: dict) -> Product:
        """Create a new product."""
        async with get_db() as db:
            product = Product(**data)
            db.add(product)
            await db.flush()
            return product
    
    async def update(self, product_id: str, data: dict) -> Optional[Product]:
        """Update a product."""
        async with get_db() as db:
            product = await self.get_by_id(product_id)
            if product:
                for key, value in data.items():
                    if hasattr(product, key):
                        setattr(product, key, value)
                await db.flush()
            return product
    
    async def delete(self, product_id: str) -> bool:
        """Delete a product."""
        async with get_db() as db:
            product = await self.get_by_id(product_id)
            if product:
                await db.delete(product)
                return True
            return False
