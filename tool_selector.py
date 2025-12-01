# tool_selector.py
from openai import OpenAI, APIError, APITimeoutError
from typing import Dict, Any, Callable
from config import config
from utils import safe_parse_json
from logger import agent_logger
from memory import conversation_memory

class ToolSelector:
    """Handles tool selection using LLM or fallback heuristics"""
    
    def __init__(self, tools: Dict[str, Callable]):
        self.tools = tools
        # Initialize OpenAI client (v1.0.0+ style)
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    def build_selector_prompt(self, examples: list = None) -> str:
        """Generate a dynamic system prompt listing available tools"""
        tool_descriptions = []
        for name, func in self.tools.items():
            doc = func.__doc__ or "No description available."
            signature = f"{name}(<input>)"
            tool_descriptions.append(f"- {signature}: {doc.strip()}")
        
        tool_list = "\n".join(tool_descriptions)
        
        prompt = (
            "You are a tool selector AI. Your job is to choose the best tool for the user's task.\n\n"
            f"Available tools:\n{tool_list}\n\n"
            "Respond ONLY with JSON in this exact format:\n"
            '{"tool":"<tool_name>","input":"<input_text>","reasoning":"<why_you_chose_this>"}\n\n'
            "Rules:\n"
            "1. Choose exactly ONE tool\n"
            "2. The 'input' should be the relevant part of the user's query\n"
            "3. Always provide reasoning for your choice\n"
        )
        
        # Add few-shot examples if provided
        if examples:
            prompt += "\nExamples:\n"
            for ex in examples:
                prompt += f"User: {ex['task']}\n"
                prompt += f"Response: {{\"tool\":\"{ex['tool']}\",\"input\":\"{ex['input']}\",\"reasoning\":\"{ex['reasoning']}\"}}\n\n"
        
        return prompt
    
    def select_with_llm(self, task: str, context: str = None) -> Dict[str, str]:
        """Use LLM to select the appropriate tool"""
        if not config.OPENAI_API_KEY:
            agent_logger.logger.warning("No OpenAI API key found, using fallback heuristics")
            return self.fallback_heuristics(task)
        
        try:
            # Build messages with optional context
            messages = [
                {"role": "system", "content": self.build_selector_prompt()}
            ]
            
            # Add conversation context if available
            if context:
                messages.append({"role": "system", "content": f"Context:\n{context}"})
            
            messages.append({"role": "user", "content": task})
            
            # Call OpenAI API (v1.0.0+ style)
            response = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages,
                max_tokens=200,
                temperature=0.0,
                timeout=config.REQUEST_TIMEOUT
            )
            
            text = response.choices[0].message.content.strip()
            selection = safe_parse_json(text)
            
            # Validate selection
            if not selection or "tool" not in selection:
                agent_logger.log_error("LLM_SELECTION", "Invalid JSON response from LLM", {"response": text})
                return self.fallback_heuristics(task)
            
            if selection["tool"] not in self.tools:
                agent_logger.log_error("LLM_SELECTION", f"LLM selected unknown tool: {selection['tool']}")
                return self.fallback_heuristics(task)
            
            agent_logger.logger.info(f"[SELECTED] LLM selected: {selection['tool']} (Reasoning: {selection.get('reasoning', 'N/A')})")
            return selection
            
        except APITimeoutError:
            agent_logger.log_error("LLM_TIMEOUT", "OpenAI API request timed out")
            return self.fallback_heuristics(task)
        except APIError as e:
            agent_logger.log_error("LLM_API_ERROR", str(e))
            return self.fallback_heuristics(task)
        except Exception as e:
            agent_logger.log_error("LLM_UNKNOWN_ERROR", str(e))
            return self.fallback_heuristics(task)
    
    def fallback_heuristics(self, task: str) -> Dict[str, str]:
        """Simple rule-based tool selection when LLM is unavailable"""
        agent_logger.logger.info("[FALLBACK] Using fallback heuristics for tool selection")
        
        t = task.lower()
        
        # Math-related keywords
        math_keywords = [
            "plus", "minus", "multiply", "divide", "add", "subtract",
            "times", "calculate", "compute", "sum", "difference",
            "+", "-", "*", "/", "**", "×", "÷", "power", "squared", "cubed"
        ]
        if any(kw in t for kw in math_keywords):
            return {
                "tool": "calculator",
                "input": task,
                "reasoning": "Detected math keywords"
            }
        
        # DateTime-related keywords
        datetime_keywords = [
            "time", "date", "day", "current", "today", "now",
            "tomorrow", "yesterday", "datetime", "clock",
            "what time", "what date", "when is"
        ]
        if any(kw in t for kw in datetime_keywords):
            return {
                "tool": "datetime",
                "input": task,
                "reasoning": "Detected datetime keywords"
            }
        
        # Wikipedia-related keywords
        wiki_keywords = [
            "wiki", "who is", "what is", "explain", "tell me about",
            "definition", "describe", "information about"
        ]
        if any(kw in t for kw in wiki_keywords):
            return {
                "tool": "wikipedia",
                "input": task,
                "reasoning": "Detected wikipedia keywords"
            }
        
        # Default to Wikipedia for general queries
        return {
            "tool": "wikipedia",
            "input": task,
            "reasoning": "Default fallback"
        }
    
    def select_tool(self, task: str, use_context: bool = True) -> Dict[str, str]:
        """Main method to select a tool (with optional conversation context)"""
        context = None
        if use_context and len(conversation_memory.history) > 0:
            context = conversation_memory.get_recent_context(n=3)
        
        return self.select_with_llm(task, context=context)