import os
import openai
import json
import re
from typing import Dict, Any
from tools import calculator, wiki_search, current_datetime

# Load OpenAI API key
#openai.api_key = os.getenv("OPENAI_API_KEY")

# Available tools
tools = {
    "calculator": calculator,
    "wikipedia": wiki_search,
    "datetime": current_datetime
}

# --- Math preprocessing ---
# --- Math preprocessing ---
def preprocess_math_expression(expression: str) -> str:
    """
    Convert natural language math expressions into valid Python expressions.
    E.g., "what is 3 minus 2?" -> "3 - 2", "5 mul by 6" -> "5 * 6"
    """
    expr = expression.lower().strip()
    
    # Remove common question words and punctuation
    expr = re.sub(r'\b(what is|what\'s|calculate|compute|solve|find|tell me)\b', '', expr, flags=re.IGNORECASE)
    expr = expr.replace('?', '').replace('!', '').strip()
    
    # Remove filler words that connect operations
    expr = re.sub(r'\b(and|then|also)\b', '', expr, flags=re.IGNORECASE)
    
    # Replace word operators with symbols
    replacements = {
        r'\bplus\b': '+',
        r'\badd\b': '+',
        r'\bminus\b': '-',
        r'\bsubtract\b': '-',
        r'\bmultiplied\s+by\b': '*',
        r'\bmul(?:tiply)?\s+by\b': '*',
        r'\bmultiply\b': '*',
        r'\btimes\b': '*',
        r'\bdivided\s+by\b': '/',
        r'\bdivide(?:d)?\s+by\b': '/',
        r'\bover\b': '/',
        r'\bto\s+the\s+power\s+of\b': '**',
        r'\bpower\s+of\b': '**',
        r'\bsquared\b': '** 2',
        r'\bcubed\b': '** 3'
    }
    
    for pattern, symbol in replacements.items():
        expr = re.sub(pattern, symbol, expr, flags=re.IGNORECASE)
    
    # Remove extra whitespace
    expr = ' '.join(expr.split())
    
    return expr
# --- JSON parsing with fallback ---
def safe_parse_json(text: str) -> Dict[str, Any]:
    """Try to parse JSON from text, fallback to empty dict on error."""
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {}

# --- Dynamic prompt generation ---
def build_selector_prompt(tools: Dict[str, Any], examples: list = []) -> str:
    """Generate a dynamic system prompt listing available tools and few-shot examples."""
    tool_descriptions = []
    for name, func in tools.items():
        doc = func.__doc__ or "No description available."
        signature = f"{name}(<input>)"
        tool_descriptions.append(f"- {signature}: {doc.strip()}")
    
    tool_list = "\n".join(tool_descriptions)

    examples_section = ""
    if examples:
        example_texts = []
        for ex in examples:
            example_texts.append(
                f"User: {ex['task']}\nAssistant: {{\"tool\":\"{ex['tool']}\",\"input\":\"{ex['input']}\"}}"
            )
        examples_section = "\n\n".join(example_texts)
    
    prompt = (
        "You are a selector that must choose exactly one tool from the following list:\n"
        f"{tool_list}\n\n"
        "Respond ONLY with JSON in this form: {\"tool\":\"<tool_name>\",\"input\":\"<text>\"} "
        "with no extra explanation.\n\n"
    )
    if examples_section:
        prompt += f"Here are some examples of correct tool selections:\n\n{examples_section}\n\n"
    return prompt

# --- Mock selector for fallback ---
def mock_llm_select_tool(task: str) -> Dict[str, str]:
    """Simple heuristic to pick a tool if LLM fails."""
    t = task.lower()
    
    # Check for math-related keywords and symbols
    if any(kw in t for kw in ["plus", "minus", "multiply", "divide", "add", "subtract", 
                               "times", "calculate", "compute", "+", "-", "*", "/", "**", "×", "÷"]):
        return {"tool": "calculator", "input": task}
    
    # Check for datetime keywords - more comprehensive list
    if any(kw in t for kw in ["time", "date", "day", "current", "today", "now", 
                               "datetime", "clock", "what time", "what date"]):
        return {"tool": "datetime", "input": task}
    
    # Check for Wikipedia keywords
    if any(kw in t for kw in ["wiki", "who is", "what is", "explain", "tell me about"]):
        return {"tool": "wikipedia", "input": task}
    
    # Default to wikipedia
    return {"tool": "wikipedia", "input": task}

# --- LLM tool selector ---
def select_tool_via_llm(task: str) -> Dict[str, str]:
    """Ask the LLM which tool to call and return {"tool": ..., "input": ...}."""
    if openai.api_key:
        prompt_system = build_selector_prompt(tools)
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": task}
                ],
                max_tokens=200,
                temperature=0.0
            )
            text = resp.choices[0].message.content.strip()
            selection = safe_parse_json(text)
            if not selection or "tool" not in selection:
                return mock_llm_select_tool(task)
            return selection
        except Exception as e:
            print(f"LLM selection failed: {e}. Using fallback heuristics.")
            return mock_llm_select_tool(task)
    else:
        return mock_llm_select_tool(task)

# --- Run agent ---
def run_agent(task: str):
    """Determine which tool to use and call it."""
    selection = select_tool_via_llm(task)
    tool_name = selection.get("tool")
    tool_input = selection.get("input", "")

    if tool_name not in tools:
        return f"Error: selected unknown tool '{tool_name}'. Available: {list(tools.keys())}"

    # Preprocess math expressions
    if tool_name == "calculator":
        tool_input = preprocess_math_expression(tool_input)

    try:
        result = tools[tool_name](tool_input)
    except Exception as e:
        result = f"Tool '{tool_name}' raised an error: {e}"
    return result

# --- CLI ---
if __name__ == "__main__":
    try:
        while True:
            user_input = input("\nWhat should the agent do? (type 'exit' to quit) ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            if not user_input:
                print("No input provided.")
                continue
            output = run_agent(user_input)
            print("\nAgent Output:\n", output)
    except KeyboardInterrupt:
        print("\nCancelled.")