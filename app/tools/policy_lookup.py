"""
Policy Lookup Tool.
Enables LLM to retrieve company policies.
"""

import logging
from typing import Any, Dict

from app.tools.registry import Tool, ToolRegistry

logger = logging.getLogger(__name__)


async def get_policy_handler(
    args: Dict[str, Any],
    session: Any = None
) -> Dict[str, Any]:
    """
    Get company policy information.
    
    Args:
        args: {
            "policy_type": str   # Type of policy to retrieve
        }
    """
    from app.db.repositories.policies import PolicyRepository
    
    policy_type = args.get("policy_type")
    
    if not policy_type:
        return {"error": "Please specify which policy you'd like to know about."}
    
    try:
        repo = PolicyRepository()
        policy = await repo.get_by_type(policy_type)
        
        if not policy:
            return {
                "policy_type": policy_type,
                "error": f"Policy '{policy_type}' not found."
            }
        
        # Get language from context
        lang = "en"
        if session:
            lang = session.context.detected_language
        
        # Use localized content if available
        title = getattr(policy, f"title_{lang}", None) or policy.title
        content = getattr(policy, f"content_{lang}", None) or policy.content
        
        return {
            "policy_type": policy_type,
            "title": title,
            "content": content,
            "effective_date": policy.effective_date.strftime("%Y-%m-%d") if policy.effective_date else None
        }
        
    except Exception as e:
        logger.error(f"Policy lookup error: {e}")
        return {"error": str(e)}


async def get_all_policies_handler(
    args: Dict[str, Any],
    session: Any = None
) -> Dict[str, Any]:
    """
    Get summary of all available policies.
    """
    from app.db.repositories.policies import PolicyRepository
    
    try:
        repo = PolicyRepository()
        policies = await repo.get_all()
        
        lang = "en"
        if session:
            lang = session.context.detected_language
        
        policy_list = []
        for policy in policies:
            title = getattr(policy, f"title_{lang}", None) or policy.title
            policy_list.append({
                "type": policy.type,
                "title": title
            })
        
        return {
            "policies": policy_list,
            "message": "Here are our available policies. Ask about any specific one for details."
        }
        
    except Exception as e:
        logger.error(f"Get all policies error: {e}")
        return {"error": str(e)}


async def search_policy_handler(
    args: Dict[str, Any],
    session: Any = None
) -> Dict[str, Any]:
    """
    Search policies by keyword.
    """
    from app.db.repositories.policies import PolicyRepository
    
    query = args.get("query", "")
    
    if not query:
        return {"error": "Please provide a search query."}
    
    try:
        repo = PolicyRepository()
        policies = await repo.search(query)
        
        if not policies:
            return {
                "query": query,
                "results": [],
                "message": "No policies found matching your query."
            }
        
        lang = "en"
        if session:
            lang = session.context.detected_language
        
        results = []
        for policy in policies:
            title = getattr(policy, f"title_{lang}", None) or policy.title
            content = getattr(policy, f"content_{lang}", None) or policy.content
            
            # Get relevant snippet
            content_lower = content.lower()
            query_lower = query.lower()
            
            idx = content_lower.find(query_lower)
            if idx != -1:
                start = max(0, idx - 50)
                end = min(len(content), idx + len(query) + 100)
                snippet = "..." + content[start:end] + "..."
            else:
                snippet = content[:150] + "..."
            
            results.append({
                "type": policy.type,
                "title": title,
                "snippet": snippet
            })
        
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Policy search error: {e}")
        return {"error": str(e)}


async def register_policy_tools(registry: ToolRegistry):
    """Register all policy-related tools."""
    
    # Get specific policy
    registry.register(Tool(
        name="get_policy",
        description="Get information about a specific company policy. Use when user asks about returns policy, shipping policy, refund policy, etc.",
        parameters={
            "policy_type": {
                "type": "string",
                "description": "Type of policy to retrieve",
                "enum": ["returns", "refunds", "shipping", "privacy", "warranty", "cancellation"]
            }
        },
        handler=get_policy_handler,
        required_params=["policy_type"]
    ))
    
    # List all policies
    registry.register(Tool(
        name="list_policies",
        description="List all available company policies. Use when user asks about what policies exist or company rules in general.",
        parameters={},
        handler=get_all_policies_handler,
        required_params=[]
    ))
    
    # Search policies
    registry.register(Tool(
        name="search_policy",
        description="Search policies for specific information. Use when user has a specific question that might be in policies.",
        parameters={
            "query": {
                "type": "string",
                "description": "Search query or question"
            }
        },
        handler=search_policy_handler,
        required_params=["query"]
    ))
    
    logger.info("Policy tools registered")
