# ...existing code...
import os
import openai
import json
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

def _parse_json_fragment(text: str) -> Dict[str, Any]:
    """Try to extract the first JSON object from text and parse it."""
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        # Final fallback: attempt direct json loads (may raise)
        return json.loads(text)

def mock_llm_select_tool(task: str) -> Dict[str, str]:
    """
    Mock LLM selector: returns a dict like {"tool": "tool_name", "input": "text"}.
    This is a simple heuristic for local testing.
    """
    t = task.lower()
    if any(op in t for op in ["+", "-", "*", "/", "**"]):
        return {"tool": "calculator", "input": task}
    if any(kw in t for kw in ["wiki", "who is", "what is", "explain"]):
        return {"tool": "wikipedia", "input": task}
    if any(kw in t for kw in ["time", "date", "day", "current"]):
        return {"tool": "datetime", "input": task}
    # Default to wikipedia for general questions in mock
    return {"tool": "wikipedia", "input": task}

def select_tool_via_llm(task: str) -> Dict[str, str]:
    """
    Ask an LLM to pick which tool to call. The LLM must respond with JSON:
    {"tool": "<calculator|wikipedia|datetime>", "input": "<text>"}
    """
    if openai.api_key:
        prompt_system = (
            "You are a selector that must choose exactly one tool from "
            '["calculator","wikipedia","datetime"] and output ONLY JSON in this form: '
            '{"tool":"<tool_name>","input":"<text>"} with no extra explanation.'
        )
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": task}
                ],
                max_tokens=200,
                temperature=0.0
            )
            text = resp.choices[0].message.content.strip()
            return _parse_json_fragment(text)
        except Exception:
            # If API fails or returns unexpected content, fall back to mock selector
            return mock_llm_select_tool(task)
    else:
        return mock_llm_select_tool(task)

def run_agent(task: str):
    """
    Ask the LLM which tool to call (via JSON). Then call the selected tool and return its result.
    """
    selection = select_tool_via_llm(task)
    if not isinstance(selection, dict):
        raise ValueError("Tool selection must be a JSON object with 'tool' and 'input' keys.")

    tool_name = selection.get("tool")
    tool_input = selection.get("input", "")

    if tool_name not in tools:
        return f"Error: selected unknown tool '{tool_name}'. Available: {list(tools.keys())}"

    # Call the chosen tool and return result
    try:
        result = tools[tool_name](tool_input)
    except Exception as e:
        result = f"Tool '{tool_name}' raised an error: {e}"
    return result

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