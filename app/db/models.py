"""
SQLAlchemy Database Models.
Defines all database entities for the voice agent.
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, Text, JSON, Boolean
from sqlalchemy.orm import relationship

from app.db.database import Base


class Product(Base):
    """Product catalog entity."""
    __tablename__ = "products"
    
    id = Column(String(50), primary_key=True)
    name = Column(String(255), nullable=False, index=True)
    name_hi = Column(String(255))  # Hindi
    name_bn = Column(String(255))  # Bengali
    name_mr = Column(String(255))  # Marathi
    
    description = Column(Text)
    description_hi = Column(Text)
    description_bn = Column(Text)
    description_mr = Column(Text)
    
    category = Column(String(100), index=True)
    price = Column(Float, nullable=False)
    stock_quantity = Column(Integer, default=0)
    image_url = Column(String(500))
    specifications = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    
    # Relationships
    faqs = relationship("FAQ", back_populates="product", lazy="selectin")
    order_items = relationship("OrderItem", back_populates="product", lazy="selectin")
    
    def __repr__(self):
        return f"<Product {self.id}: {self.name}>"


class Order(Base):
    """Customer order entity."""
    __tablename__ = "orders"
    
    id = Column(String(50), primary_key=True)
    user_phone = Column(String(20), index=True)
    status = Column(String(50), default="pending", index=True)
    total_amount = Column(Float)
    shipping_address = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    estimated_delivery = Column(DateTime)
    
    tracking_number = Column(String(100))
    shipping_carrier = Column(String(100))
    
    # Relationships
    items = relationship("OrderItem", back_populates="order", lazy="selectin")
    returns = relationship("Return", back_populates="order", lazy="selectin")
    
    def __repr__(self):
        return f"<Order {self.id}: {self.status}>"


class OrderItem(Base):
    """Order line item entity."""
    __tablename__ = "order_items"
    
    id = Column(String(50), primary_key=True)
    order_id = Column(String(50), ForeignKey("orders.id"), index=True)
    product_id = Column(String(50), ForeignKey("products.id"), index=True)
    quantity = Column(Integer, default=1)
    unit_price = Column(Float)
    
    # Relationships
    order = relationship("Order", back_populates="items")
    product = relationship("Product", back_populates="order_items")
    
    def __repr__(self):
        return f"<OrderItem {self.id}: {self.product_id} x{self.quantity}>"


class FAQ(Base):
    """Frequently Asked Questions entity."""
    __tablename__ = "faqs"
    
    id = Column(String(50), primary_key=True)
    product_id = Column(String(50), ForeignKey("products.id"), nullable=True, index=True)
    
    question = Column(Text, nullable=False)
    question_hi = Column(Text)
    question_bn = Column(Text)
    question_mr = Column(Text)
    
    answer = Column(Text, nullable=False)
    answer_hi = Column(Text)
    answer_bn = Column(Text)
    answer_mr = Column(Text)
    
    topic = Column(String(100), index=True)  # warranty, shipping, returns, usage, specifications
    is_general = Column(Boolean, default=False)  # True if not product-specific
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    product = relationship("Product", back_populates="faqs")
    
    def __repr__(self):
        return f"<FAQ {self.id}: {self.topic}>"


class Policy(Base):
    """Company policy entity."""
    __tablename__ = "policies"
    
    id = Column(String(50), primary_key=True)
    type = Column(String(50), unique=True, index=True)  # returns, refunds, shipping, privacy, warranty
    
    title = Column(String(255))
    title_hi = Column(String(255))
    title_bn = Column(String(255))
    title_mr = Column(String(255))
    
    content = Column(Text)
    content_hi = Column(Text)
    content_bn = Column(Text)
    content_mr = Column(Text)
    
    effective_date = Column(DateTime)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Policy {self.type}>"


class Return(Base):
    """Return request entity."""
    __tablename__ = "returns"
    
    id = Column(String(50), primary_key=True)
    order_id = Column(String(50), ForeignKey("orders.id"), index=True)
    
    reason = Column(String(100))
    action_requested = Column(String(50))  # refund, exchange, store_credit
    status = Column(String(50), default="pending")  # pending, approved, rejected, completed
    
    refund_amount = Column(Float)
    refund_method = Column(String(50))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    
    # Relationships
    order = relationship("Order", back_populates="returns")
    
    def __repr__(self):
        return f"<Return {self.id}: {self.status}>"


class Conversation(Base):
    """Conversation session entity for persistence."""
    __tablename__ = "conversations"
    
    id = Column(String(50), primary_key=True)
    session_id = Column(String(100), unique=True, index=True)
    user_phone = Column(String(20), index=True)
    
    turns = Column(JSON)  # Serialized conversation turns
    context = Column(JSON)  # Serialized context state
    
    language = Column(String(10), default="en")
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Conversation {self.session_id}>"
