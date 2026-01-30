"""
Product Search Tool.
Enables LLM to search products by name, category, or description.
"""

import logging
from typing import Any, Dict, List, Optional

from app.tools.registry import Tool, ToolRegistry

logger = logging.getLogger(__name__)


async def search_products_handler(
    args: Dict[str, Any],
    session: Any = None
) -> Dict[str, Any]:
    """
    Search for products in the catalog.
    
    Args:
        args: {
            "query": str,           # Search query
            "category": str,        # Optional category filter
            "max_results": int,     # Optional limit (default 5)
            "price_min": float,     # Optional minimum price
            "price_max": float      # Optional maximum price
        }
        session: User session for context
    
    Returns:
        {
            "products": [...],
            "total_count": int,
            "query": str
        }
    """
    from app.db.repositories.products import ProductRepository
    
    query = args.get("query", "")
    category = args.get("category")
    max_results = args.get("max_results", 5)
    price_min = args.get("price_min")
    price_max = args.get("price_max")
    
    try:
        repo = ProductRepository()
        
        products = await repo.search(
            query=query,
            category=category,
            price_min=price_min,
            price_max=price_max,
            limit=max_results
        )
        
        # Format results for LLM
        formatted_products = []
        for p in products:
            formatted_products.append({
                "id": p.id,
                "name": p.name,
                "description": p.description[:100] + "..." if len(p.description) > 100 else p.description,
                "category": p.category,
                "price": p.price,
                "in_stock": p.stock_quantity > 0
            })
        
        # Update session context if available
        if session and formatted_products:
            session.context.last_product_id = formatted_products[0]["id"]
            session.context.last_product_name = formatted_products[0]["name"]
            session.context.last_search_query = query
        
        return {
            "products": formatted_products,
            "total_count": len(formatted_products),
            "query": query
        }
        
    except Exception as e:
        logger.error(f"Product search error: {e}")
        return {
            "products": [],
            "total_count": 0,
            "query": query,
            "error": str(e)
        }


async def get_product_details_handler(
    args: Dict[str, Any],
    session: Any = None
) -> Dict[str, Any]:
    """
    Get detailed information about a specific product.
    
    Args:
        args: {
            "product_id": str   # Product ID to look up
        }
        session: User session for context
    
    Returns:
        Product details or error message
    """
    from app.db.repositories.products import ProductRepository
    
    product_id = args.get("product_id")
    
    if not product_id:
        # Try to use last product from context
        if session and session.context.last_product_id:
            product_id = session.context.last_product_id
        else:
            return {"error": "No product ID provided"}
    
    try:
        repo = ProductRepository()
        product = await repo.get_by_id(product_id)
        
        if not product:
            return {"error": f"Product '{product_id}' not found"}
        
        # Update session context
        if session:
            session.context.last_product_id = product.id
            session.context.last_product_name = product.name
        
        return {
            "id": product.id,
            "name": product.name,
            "description": product.description,
            "category": product.category,
            "price": product.price,
            "stock_quantity": product.stock_quantity,
            "in_stock": product.stock_quantity > 0,
            "specifications": product.specifications or {}
        }
        
    except Exception as e:
        logger.error(f"Get product error: {e}")
        return {"error": str(e)}


async def get_products_by_category_handler(
    args: Dict[str, Any],
    session: Any = None
) -> Dict[str, Any]:
    """
    Get products in a specific category.
    """
    from app.db.repositories.products import ProductRepository
    
    category = args.get("category")
    limit = args.get("limit", 5)
    
    if not category:
        return {"error": "No category provided"}
    
    try:
        repo = ProductRepository()
        products = await repo.get_by_category(category, limit=limit)
        
        formatted = [
            {
                "id": p.id,
                "name": p.name,
                "price": p.price,
                "in_stock": p.stock_quantity > 0
            }
            for p in products
        ]
        
        if session:
            session.context.last_category = category
        
        return {
            "category": category,
            "products": formatted,
            "count": len(formatted)
        }
        
    except Exception as e:
        logger.error(f"Get by category error: {e}")
        return {"error": str(e)}


async def register_product_tools(registry: ToolRegistry):
    """Register all product-related tools."""
    
    # Search products
    registry.register(Tool(
        name="search_products",
        description="Search for products by name, description, or keywords. Use this when the user asks about finding or looking for products.",
        parameters={
            "query": {
                "type": "string",
                "description": "Search query - product name, keywords, or description"
            },
            "category": {
                "type": "string",
                "description": "Filter by category",
                "enum": ["electronics", "clothing", "home", "beauty", "sports", "books", "toys", "grocery"]
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 5
            },
            "price_min": {
                "type": "number",
                "description": "Minimum price filter"
            },
            "price_max": {
                "type": "number",
                "description": "Maximum price filter"
            }
        },
        handler=search_products_handler,
        required_params=["query"]
    ))
    
    # Get product details
    registry.register(Tool(
        name="get_product_details",
        description="Get detailed information about a specific product including price, description, specifications, and availability. Use when user asks for more details about a product.",
        parameters={
            "product_id": {
                "type": "string",
                "description": "The product ID to get details for. If not provided, uses the last discussed product."
            }
        },
        handler=get_product_details_handler,
        required_params=[]
    ))
    
    # Get products by category
    registry.register(Tool(
        name="get_products_by_category",
        description="List products in a specific category. Use when user asks to see products in a category.",
        parameters={
            "category": {
                "type": "string",
                "description": "Product category to browse",
                "enum": ["electronics", "clothing", "home", "beauty", "sports", "books", "toys", "grocery"]
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of products to return",
                "default": 5
            }
        },
        handler=get_products_by_category_handler,
        required_params=["category"]
    ))
    
    logger.info("Product tools registered")
