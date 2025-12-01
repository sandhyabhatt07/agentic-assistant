# memory.py
from typing import List, Dict, Any
from datetime import datetime
from logger import agent_logger

class ConversationMemory:
    """Stores conversation history and context"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}
    
    def add_interaction(self, user_input: str, tool_used: str, tool_input: str, result: str):
        """Add an interaction to memory"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "tool": tool_used,
            "tool_input": tool_input,
            "result": result
        }
        
        self.history.append(interaction)
        
        # Keep only recent history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        agent_logger.logger.debug(f"Added interaction to memory (history size: {len(self.history)})")
    
    def get_recent_context(self, n: int = 3) -> str:
        """Get formatted recent context for LLM"""
        if not self.history:
            return "No previous interactions."
        
        recent = self.history[-n:]
        context_lines = ["Previous interactions:"]
        
        for i, interaction in enumerate(recent, 1):
            context_lines.append(
                f"{i}. User asked: '{interaction['user_input']}' → "
                f"Used {interaction['tool']} → Result: {interaction['result'][:100]}"
            )
        
        return "\n".join(context_lines)
    
    def store_variable(self, key: str, value: Any):
        """Store a variable in context (e.g., 'X = 5')"""
        self.context[key] = value
        agent_logger.logger.debug(f"Stored variable: {key} = {value}")
    
    def get_variable(self, key: str) -> Any:
        """Retrieve a stored variable"""
        return self.context.get(key)
    
    def clear(self):
        """Clear all memory"""
        self.history.clear()
        self.context.clear()
        agent_logger.logger.info("Memory cleared")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get memory summary"""
        return {
            "total_interactions": len(self.history),
            "stored_variables": len(self.context),
            "recent_tools_used": [h["tool"] for h in self.history[-5:]]
        }

# Global memory instance
conversation_memory = ConversationMemory()