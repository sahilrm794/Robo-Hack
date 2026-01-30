"""Database repositories initialization."""

from app.db.repositories.products import ProductRepository
from app.db.repositories.orders import OrderRepository
from app.db.repositories.faqs import FAQRepository
from app.db.repositories.policies import PolicyRepository
from app.db.repositories.returns import ReturnRepository

__all__ = [
    "ProductRepository",
    "OrderRepository",
    "FAQRepository",
    "PolicyRepository",
    "ReturnRepository"
]
