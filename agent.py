# agent.py (Enhanced with better multi-tool support)
import time
import json
from typing import Dict, Callable, Any, Optional, List
from tools import calculator, wiki_search, current_datetime
from tool_selector import ToolSelector
from memory import conversation_memory
from logger import agent_logger
from config import config
from utils import preprocess_math_expression
from openai import OpenAI

class Agent:
    """Main agent that orchestrates tool selection and execution"""
    
    def __init__(self):
        # Register tools
        self.tools: Dict[str, Callable] = {
            "calculator": calculator,
            "wikipedia": wiki_search,
            "datetime": current_datetime
        }
        
        # Initialize tool selector
        self.selector = ToolSelector(self.tools)
        
        # Initialize OpenAI client for conversation handling
        self.client = OpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None
        
        # Validate configuration
        config.validate()
        
        agent_logger.logger.info("Agent initialized with tools: " + ", ".join(self.tools.keys()))
    
    def decompose_question(self, task: str) -> List[Dict[str, str]]:
        """
        Decompose a multi-part question into separate sub-tasks
        Returns: List of {description, query} dictionaries
        """
        if not self.client:
            return [{"description": task, "query": task}]
        
        system_prompt = (
            "You are a question decomposer. Break down multi-part questions into separate sub-tasks.\n"
            "Each sub-task should be answerable by a single tool (calculator, wikipedia, or datetime).\n\n"
            "Respond with JSON array:\n"
            '[{"description": "calculate 5+3", "query": "5+3"}, '
            '{"description": "get info about Python", "query": "Python programming language"}]\n\n'
            "If it's a single question, return an array with one item."
        )
        
        try:
            response = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Break down this question: {task}"}
                ],
                max_tokens=300,
                temperature=0.0
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            return result if isinstance(result, list) else [{"description": task, "query": task}]
            
        except Exception as e:
            agent_logger.logger.warning(f"Error decomposing question: {e}")
            return [{"description": task, "query": task}]
    
    def needs_tool(self, task: str, context: str = None) -> tuple[bool, Optional[str]]:
        """
        Determine if the task needs a tool or can be answered directly.
        Returns: (needs_tool: bool, direct_response: Optional[str])
        """
        if not self.client:
            return True, None  # Always use tools if no LLM available
        
        # Build context-aware prompt
        system_prompt = (
            "You are an assistant that decides if a user query needs a tool or can be answered directly.\n"
            "Available tools: calculator, wikipedia, datetime\n\n"
            "IMPORTANT: \n"
            "- If the user asks MULTIPLE questions, and at least one needs a tool, set needs_tool to true\n"
            "- If asking for confirmation/verification about previous results, you can answer directly using context\n"
            "- For multi-part questions (e.g., 'are you sure? what about X?'), focus on the part that needs a tool\n\n"
            "Respond with JSON:\n"
            '{"needs_tool": true/false, "response": "your direct answer if needs_tool is false"}\n\n'
            "If needs_tool is false, provide a helpful direct response using conversation context."
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation context
        if context:
            messages.append({"role": "system", "content": f"Conversation history:\n{context}"})
        
        messages.append({"role": "user", "content": task})
        
        try:
            response = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages,
                max_tokens=150,
                temperature=0.0
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            
            if not result.get("needs_tool", True):
                return False, result.get("response", "I'm not sure about that.")
            
            return True, None
            
        except Exception as e:
            agent_logger.logger.warning(f"Error in needs_tool check: {e}, defaulting to tool usage")
            return True, None
    
    def preprocess_input(self, tool_name: str, tool_input: str) -> str:
        """Preprocess input based on tool type"""
        if tool_name == "calculator":
            return preprocess_math_expression(tool_input)
        return tool_input
    
    def execute_tool(self, tool_name: str, tool_input: str) -> tuple[str, bool]:
        """Execute a tool and return (result, success)"""
        if tool_name not in self.tools:
            error_msg = f"Unknown tool: '{tool_name}'. Available: {list(self.tools.keys())}"
            return error_msg, False
        
        try:
            # Preprocess input if needed
            processed_input = self.preprocess_input(tool_name, tool_input)
            
            # Execute tool
            start_time = time.time()
            result = self.tools[tool_name](processed_input)
            duration = time.time() - start_time
            
            # Check if result indicates an error
            success = not str(result).startswith("Error:")
            
            # Log the execution
            agent_logger.log_tool_call(
                tool_name=tool_name,
                input_data=processed_input,
                output=str(result),
                duration=duration,
                success=success
            )
            
            return str(result), success
            
        except Exception as e:
            error_msg = f"Tool '{tool_name}' raised an exception: {str(e)}"
            agent_logger.log_error("TOOL_EXECUTION", error_msg, {"tool": tool_name, "input": tool_input})
            return error_msg, False
    
    def run(self, task: str, use_memory: bool = True) -> str:
        """
        Main method to run the agent on a task
        
        Args:
            task: User's task/query
            use_memory: Whether to use conversation context
        
        Returns:
            Result string
        """
        agent_logger.log_agent_start(task)
        start_time = time.time()
        
        # Get conversation context
        context = None
        if use_memory and len(conversation_memory.history) > 0:
            context = conversation_memory.get_recent_context(n=3)
        
        # Step 1: Check if we need a tool or can answer directly
        needs_tool, direct_response = self.needs_tool(task, context)
        
        if not needs_tool and direct_response:
            agent_logger.logger.info(f"[DIRECT] Answering without tool: {direct_response[:50]}...")
            
            # Store in memory
            if use_memory:
                conversation_memory.add_interaction(
                    user_input=task,
                    tool_used="none",
                    tool_input="",
                    result=direct_response
                )
            
            total_duration = time.time() - start_time
            agent_logger.log_agent_complete(task, direct_response, total_duration)
            return direct_response
        
        # Step 2: Select and execute tool
        selection = self.selector.select_tool(task, use_context=use_memory)
        tool_name = selection.get("tool")
        tool_input = selection.get("input", task)
        
        # Step 3: Execute tool
        result, success = self.execute_tool(tool_name, tool_input)
        
        # Step 4: Store in memory
        if use_memory:
            conversation_memory.add_interaction(
                user_input=task,
                tool_used=tool_name,
                tool_input=tool_input,
                result=result
            )
        
        # Step 5: Log completion
        total_duration = time.time() - start_time
        agent_logger.log_agent_complete(task, result, total_duration)
        
        return result
    
    def run_with_retry(self, task: str, max_retries: int = None) -> str:
        """Run agent with retry logic on failure"""
        max_retries = max_retries or config.MAX_RETRIES
        
        for attempt in range(1, max_retries + 1):
            result = self.run(task)
            
            # Check if successful
            if not result.startswith("Error:") and not result.startswith("Tool"):
                return result
            
            if attempt < max_retries:
                agent_logger.logger.warning(f"Attempt {attempt} failed, retrying...")
                time.sleep(0.5 * attempt)
        
        return f"Failed after {max_retries} attempts. Last error: {result}"
    
    def add_tool(self, name: str, func: Callable, description: str = None):
        """Dynamically add a new tool"""
        if description:
            func.__doc__ = description
        self.tools[name] = func
        self.selector.tools = self.tools
        agent_logger.logger.info(f"Added new tool: {name}")
    
    def remove_tool(self, name: str):
        """Remove a tool"""
        if name in self.tools:
            del self.tools[name]
            self.selector.tools = self.tools
            agent_logger.logger.info(f"Removed tool: {name}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            "available_tools": list(self.tools.keys()),
            "memory": conversation_memory.get_summary(),
            "session": agent_logger.get_session_stats()
        }
    
    def reset(self):
        """Reset agent state"""
        conversation_memory.clear()
        agent_logger.logger.info("Agent reset")

    def run_multi_tool(self, task: str, max_iterations: int = 5) -> str:
        """
        Run agent with ability to use multiple tools in sequence
        Improved version that decomposes the question first
        """
        agent_logger.log_agent_start(f"[MULTI-TOOL] {task}")
        start_time = time.time()
        
        print(f"\n🤖 Decomposing multi-part question...\n")
        
        # Step 1: Decompose the question
        sub_tasks = self.decompose_question(task)
        
        if len(sub_tasks) <= 1:
            # Not actually multi-part, use regular run
            return self.run(task)
        
        print(f"📋 Found {len(sub_tasks)} sub-tasks:\n")
        
        # Step 2: Execute each sub-task
        results = []
        for i, sub_task in enumerate(sub_tasks, 1):
            print(f"  {i}. {sub_task['description']}")
            
            # Select and execute tool for this sub-task
            selection = self.selector.select_tool(sub_task['query'], use_context=True)
            tool_name = selection.get("tool")
            tool_input = selection.get("input", sub_task['query'])
            
            print(f"     → Using {tool_name}: {tool_input}")
            
            result, success = self.execute_tool(tool_name, tool_input)
            
            results.append({
                "task": sub_task['description'],
                "tool": tool_name,
                "result": result,
                "success": success
            })
            
            print(f"     ✓ {result[:80]}...\n")
        
        # Step 3: Combine results
        final_answer = self._synthesize_results(task, results)
        
        # Step 4: Store in memory
        conversation_memory.add_interaction(
            user_input=task,
            tool_used="multi-tool",
            tool_input=str([r["task"] for r in results]),
            result=final_answer
        )
        
        total_duration = time.time() - start_time
        agent_logger.log_agent_complete(task, final_answer, total_duration)
        
        return final_answer
    
    def _synthesize_results(self, original_task: str, results: List[Dict]) -> str:
        """
        Synthesize multiple tool results into a coherent final answer
        """
        if not self.client:
            # Fallback: just concatenate results
            return self._combine_results_simple(results)
        
        # Use LLM to create a natural response
        system_prompt = (
            "You are synthesizing multiple tool results into a single coherent answer.\n"
            "Combine the results naturally to answer the user's original question.\n"
            "Be concise but complete."
        )
        
        results_text = "\n".join([
            f"- {r['task']}: {r['result']}"
            for r in results if r['success']
        ])
        
        try:
            response = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Original question: {original_task}\n\nResults:\n{results_text}\n\nProvide a natural answer:"}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            agent_logger.logger.warning(f"Error synthesizing results: {e}")
            return self._combine_results_simple(results)
    
    def _combine_results_simple(self, results: List[Dict]) -> str:
        """Simple fallback for combining results"""
        successful = [r for r in results if r["success"]]
        
        if not successful:
            return "All tool calls failed."
        
        combined = "\n\n".join([
            f"{r['task']}: {r['result']}" 
            for r in successful
        ])
        
        return combined