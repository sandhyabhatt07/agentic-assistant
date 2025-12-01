# Agentic Assistant — Minimal GPT-based Agent

A minimal agentic AI assistant in Python that reads a user question, decides which tool should handle it (via GPT or a mock selector), executes the tool, and returns the result. The project demonstrates dynamic prompt construction, few-shot guidance for tool selection, validated JSON outputs from the LLM, tool execution, fallback heuristics, and simple memory/log handling.

---

## Features

- GPT-driven tool selection with few-shot examples and dynamic prompt generation
- Validation of LLM output (expects JSON: `{"tool":"<name>","input":"<text>"}`)
- Fallback to local heuristics/mock selector when OpenAI API is unavailable
- Tool implementations for:
  - Calculator (safe math evaluation)
  - Wikipedia search (summary)
  - Date/time utilities (natural-language parsing with fallback)
- Simple CLI loop to interact with the agent
- Modular structure to add new tools, change selectors, or swap LLM providers
- Logging and optional memory support (conversation/usage)

---

## Project Structure

agentic-assistant/
```
├── app.py               # (simple GPT-based agent / older variant)
├── app1.py              # keyword-based reference agent
├── app2.py              # full GPT-based selector + CLI (example advanced)
├── tools.py             # calculator, wiki_search, current_datetime
├── tool_selector.py     # prompt builder and LLM selection logic (optional split)
├── agent.py             # run_agent wrapper that orchestrates selection + execution
├── main.py              # simple entrypoint/CLI that calls agent.run_agent
├── config.py            # configuration (models, API keys defaults)
├── logger.py            # logging helpers
├── memory.py            # simple in-memory (or file) convo memory
├── utils.py             # helpers (json parsing, safe-eval, prompt formatting)
├── README.md            # this file
└── requirements.txt     # Python dependencies
```

---

## Modules Overview

- tools.py
  - calculator(expression): evaluate arithmetic expressions (use safe eval).
  - wiki_search(query): returns a short Wikipedia summary (uses `wikipedia` package).
  - current_datetime(query): interprets natural language date/time queries (LLM-assisted with keyword fallback).

- tool_selector.py
  - Build dynamic system prompts listing available tools and few-shot examples.
  - Sends a prompt to the LLM and expects JSON output selecting a tool and input.
  - Validates and sanitizes LLM output and falls back to a mock selector when necessary.

- agent.py
  - High-level orchestration: accepts a user task, calls selector, calls chosen tool, returns result.

- main.py
  - CLI entrypoint that loops, reads user input, calls run_agent, prints output. Stops on `exit`/`quit`.

- config.py
  - Central place for configuration values (e.g., default model, timeouts, environment var names).

- logger.py
  - Simple logging wrapper to write logs to console/file with levels.

- memory.py
  - Optional short-term memory (e.g., last N interactions) for context reuse.

- utils.py
  - Helpers: robust JSON fragment parsing, safe expression parsing/normalization, prompt formatting.

- requirements.txt
  - Libraries required (e.g., openai, wikipedia). See the Setup section.

---

## Setup

Requirements
- Python 3.10+ recommended
- Internet for OpenAI / Wikipedia calls (optional: agent works with mock offline)

pip install -r [requirements.txt](http://_vscodecontentref_/0)

# Extending the Agent

Add a New Tool:

def weather_lookup(city: str):
    return f"Weather for {city} (placeholder)"

agent.add_tool(
    name="weather",
    func=weather_lookup,
    description="Returns weather information for a city"
)

The tool will automatically appear in the LLM's prompt and be available for selection.

This project was built as a learning experiment with GitHub Copilot and GPT-based tool reasoning.