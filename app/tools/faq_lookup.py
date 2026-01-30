"""
FAQ Lookup Tool.
Enables LLM to search and retrieve FAQ answers.
"""

import logging
from typing import Any, Dict, List

from app.tools.registry import Tool, ToolRegistry

logger = logging.getLogger(__name__)


async def get_product_faq_handler(
    args: Dict[str, Any],
    session: Any = None
) -> Dict[str, Any]:
    """
    Get FAQs for a specific product.
    
    Args:
        args: {
            "product_id": str,      # Product to get FAQs for
            "question_topic": str   # Optional topic filter
        }
    """
    from app.db.repositories.faqs import FAQRepository
    
    product_id = args.get("product_id")
    topic = args.get("question_topic")
    
    # Use last product from context if not provided
    if not product_id and session and session.context.last_product_id:
        product_id = session.context.last_product_id
    
    if not product_id:
        return {"error": "No product specified. Please specify a product first."}
    
    try:
        repo = FAQRepository()
        faqs = await repo.get_for_product(product_id, topic=topic)
        
        if not faqs:
            return {
                "product_id": product_id,
                "faqs": [],
                "message": "No FAQs found for this product."
            }
        
        # Get language from context
        lang = "en"
        if session:
            lang = session.context.detected_language
        
        formatted_faqs = []
        for faq in faqs:
            # Use localized content if available
            question = getattr(faq, f"question_{lang}", None) or faq.question
            answer = getattr(faq, f"answer_{lang}", None) or faq.answer
            
            formatted_faqs.append({
                "question": question,
                "answer": answer,
                "topic": faq.topic
            })
        
        return {
            "product_id": product_id,
            "faqs": formatted_faqs,
            "count": len(formatted_faqs)
        }
        
    except Exception as e:
        logger.error(f"FAQ lookup error: {e}")
        return {"error": str(e)}


async def search_faq_handler(
    args: Dict[str, Any],
    session: Any = None
) -> Dict[str, Any]:
    """
    Search FAQs by question or keyword.
    
    Args:
        args: {
            "query": str,       # Search query
            "topic": str        # Optional topic filter
        }
    """
    from app.db.repositories.faqs import FAQRepository
    
    query = args.get("query", "")
    topic = args.get("topic")
    
    try:
        repo = FAQRepository()
        faqs = await repo.search(query=query, topic=topic)
        
        if not faqs:
            return {
                "query": query,
                "faqs": [],
                "message": "No matching FAQs found."
            }
        
        # Get language from context
        lang = "en"
        if session:
            lang = session.context.detected_language
        
        formatted_faqs = []
        for faq in faqs[:5]:  # Limit to 5 results
            question = getattr(faq, f"question_{lang}", None) or faq.question
            answer = getattr(faq, f"answer_{lang}", None) or faq.answer
            
            formatted_faqs.append({
                "question": question,
                "answer": answer,
                "topic": faq.topic,
                "product_id": faq.product_id
            })
        
        return {
            "query": query,
            "faqs": formatted_faqs,
            "count": len(formatted_faqs)
        }
        
    except Exception as e:
        logger.error(f"FAQ search error: {e}")
        return {"error": str(e)}


async def get_general_faq_handler(
    args: Dict[str, Any],
    session: Any = None
) -> Dict[str, Any]:
    """
    Get general (non-product specific) FAQs.
    
    Args:
        args: {
            "topic": str    # FAQ topic (shipping, returns, payment, etc.)
        }
    """
    from app.db.repositories.faqs import FAQRepository
    
    topic = args.get("topic", "general")
    
    try:
        repo = FAQRepository()
        faqs = await repo.get_general(topic=topic)
        
        lang = "en"
        if session:
            lang = session.context.detected_language
        
        formatted_faqs = []
        for faq in faqs:
            question = getattr(faq, f"question_{lang}", None) or faq.question
            answer = getattr(faq, f"answer_{lang}", None) or faq.answer
            
            formatted_faqs.append({
                "question": question,
                "answer": answer
            })
        
        return {
            "topic": topic,
            "faqs": formatted_faqs,
            "count": len(formatted_faqs)
        }
        
    except Exception as e:
        logger.error(f"General FAQ error: {e}")
        return {"error": str(e)}


async def register_faq_tools(registry: ToolRegistry):
    """Register all FAQ-related tools."""
    
    # Get product FAQs
    registry.register(Tool(
        name="get_product_faq",
        description="Get frequently asked questions about a specific product. Use when user asks questions about a product's warranty, usage, specifications, etc.",
        parameters={
            "product_id": {
                "type": "string",
                "description": "Product ID to get FAQs for. If not provided, uses the last discussed product."
            },
            "question_topic": {
                "type": "string",
                "description": "Filter by topic",
                "enum": ["warranty", "shipping", "returns", "usage", "specifications", "compatibility"]
            }
        },
        handler=get_product_faq_handler,
        required_params=[]
    ))
    
    # Search FAQs
    registry.register(Tool(
        name="search_faq",
        description="Search all FAQs by question or keyword. Use when user has a general question that might be in FAQs.",
        parameters={
            "query": {
                "type": "string",
                "description": "Search query or question keywords"
            },
            "topic": {
                "type": "string",
                "description": "Optional topic filter",
                "enum": ["warranty", "shipping", "returns", "payment", "account", "general"]
            }
        },
        handler=search_faq_handler,
        required_params=["query"]
    ))
    
    # Get general FAQs
    registry.register(Tool(
        name="get_general_faq",
        description="Get general FAQs about shipping, payment, account, etc. Use for non-product specific questions.",
        parameters={
            "topic": {
                "type": "string",
                "description": "FAQ topic",
                "enum": ["shipping", "returns", "payment", "account", "general"]
            }
        },
        handler=get_general_faq_handler,
        required_params=["topic"]
    ))
    
    logger.info("FAQ tools registered")
