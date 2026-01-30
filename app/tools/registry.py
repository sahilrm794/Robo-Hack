"""
Tool Registry and Executor.
Manages tool registration and execution for LLM function calling.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Awaitable
from functools import wraps

from app.config import get_settings
from app.core.exceptions import ToolNotFoundException, ToolExecutionException, ToolValidationException

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class Tool:
    """Tool definition for LLM function calling."""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Awaitable[Dict[str, Any]]]
    required_params: List[str] = field(default_factory=list)
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI/Groq tool schema format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required_params
                }
            }
        }


class ToolRegistry:
    """
    Registry for managing and executing tools.
    
    Tools are registered with their schemas and handlers.
    The registry provides tool schemas to the LLM and executes tool calls.
    """
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._db = None
    
    async def initialize(self):
        """Initialize the tool registry with all available tools."""
        logger.info("Initializing tool registry...")
        
        # Import and register all tools
        from app.tools.product_search import register_product_tools
        from app.tools.faq_lookup import register_faq_tools
        from app.tools.order_tracking import register_order_tools
        from app.tools.returns import register_return_tools
        from app.tools.policy_lookup import register_policy_tools
        
        # Register each tool category
        await register_product_tools(self)
        await register_faq_tools(self)
        await register_order_tools(self)
        await register_return_tools(self)
        await register_policy_tools(self)
        
        logger.info(f"Registered {len(self._tools)} tools: {list(self._tools.keys())}")
    
    def register(self, tool: Tool):
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get all tool schemas for LLM."""
        return [tool.to_schema() for tool in self._tools.values()]
    
    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        session: Any = None
    ) -> Dict[str, Any]:
        """
        Execute a tool with given arguments.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            session: Optional session for context
        
        Returns:
            Tool execution result
        """
        tool = self._tools.get(tool_name)
        
        if not tool:
            raise ToolNotFoundException(tool_name)
        
        # Validate required parameters
        missing = [p for p in tool.required_params if p not in arguments]
        if missing:
            raise ToolValidationException(tool_name, [f"Missing required parameter: {p}" for p in missing])
        
        start_time = time.time()
        
        try:
            # Execute tool handler
            result = await tool.handler(arguments, session)
            
            execution_time = (time.time() - start_time) * 1000
            logger.info(f"Tool {tool_name} executed in {execution_time:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Tool execution error: {tool_name} - {e}")
            raise ToolExecutionException(tool_name, str(e))


def tool(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    required: Optional[List[str]] = None
):
    """
    Decorator for registering tool handlers.
    
    Usage:
        @tool(
            name="search_products",
            description="Search for products",
            parameters={...},
            required=["query"]
        )
        async def search_products(args, session):
            ...
    """
    def decorator(func: Callable[..., Awaitable[Dict[str, Any]]]):
        @wraps(func)
        async def wrapper(args: Dict[str, Any], session: Any = None):
            return await func(args, session)
        
        wrapper._tool_info = Tool(
            name=name,
            description=description,
            parameters=parameters,
            handler=wrapper,
            required_params=required or []
        )
        
        return wrapper
    
    return decorator
