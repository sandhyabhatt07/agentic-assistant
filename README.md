# Agentic Assistant — Multi-Tool GPT Agent

An intelligent AI assistant that handles single and multi-part questions by automatically selecting and orchestrating multiple tools. Built with GPT-3.5 Turbo for task decomposition, tool selection, and result synthesis.

---

## Features

- **Multi-Part Question Handling**: Automatically detects and answers complex questions requiring multiple tools
- **Smart Tool Selection**: GPT-driven selection with few-shot learning and JSON validation
- **Three Core Tools**: Calculator, Wikipedia search, DateTime utilities
- **Conversation Memory**: Context-aware responses using interaction history
- **Interactive CLI**: Commands for stats, memory, export, and more

---

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/yourusername/agentic-assistant.git
cd agentic-assistant
pip install -r requirements.txt

# Set API key
echo "OPENAI_API_KEY=your-key" > .env

# Run
python main.py
```

### Usage Examples

```bash
# Multi-part questions
You: what is 5 + 3 and also tell me about Python
Agent: 5 + 3 equals 8. Python is a high-level programming language...

# Single questions
You: calculate 25 * 4
Agent: 100

You: tell me about Einstein
Agent: Albert Einstein was a theoretical physicist...

# Context-aware
You: what is 10 * 5?
Agent: 50
You: multiply that by 2
Agent: 100

# Commands
You: stats      # Show statistics
You: memory     # View history
You: clear      # Reset memory
You: export     # Save logs
```

---

## How It Works

**Single Question:**
```
"what is 25 * 4?" → Tool Selector → Calculator → "100"
```

**Multi-Part Question:**
```
"what is 5+3 and tell me about Python"
  ↓
Decompose: ["5+3", "Python programming"]
  ↓
Execute: calculator(5+3) → 8
         wikipedia(Python) → article
  ↓
Synthesize → "5+3 equals 8. Python is..."
```

---

## Project Structure

```
agentic-assistant/
├── main.py              # CLI with multi-part detection
├── agent.py             # Core orchestrator with multi-tool execution
├── tool_selector.py     # GPT-based tool selection
├── tools.py             # Calculator, Wikipedia, DateTime
├── config.py            # Configuration (model: gpt-3.5-turbo)
├── logger.py            # Session logging and stats
├── memory.py            # Conversation history
└── utils.py             # Helper functions
```

---

## Testing Scenarios

**Multi-Part:**
- `what is 5+3 and also tell me about Python`
- `calculate 25*4 and search for Einstein`
- `what's the time and what is 100/5?`

**Single Tool:**
- `what is 50 + 25?` (calculator)
- `tell me about AI` (wikipedia)
- `what's the date?` (datetime)

**Context:**
- `what is 10 * 5?` → `multiply that by 2`

**Edge Cases:**
- `what is Python?` (ambiguous)
- `divide 100 by 0` (error handling)
- `asdfghjkl` (invalid input)

---

## Configuration

Edit `config.py`:

```python
OPENAI_MODEL = "gpt-3.5-turbo"  # Model used
MAX_RETRIES = 3                  # Retry attempts
MAX_MEMORY_SIZE = 50             # Conversation history size
```

---

## Adding Tools

```python
def weather_lookup(city: str):
    return f"Weather for {city}: Sunny, 72°F"

agent.add_tool(
    name="weather",
    func=weather_lookup,
    description="Get weather for a city"
)
```

Tool automatically appears in prompts and multi-tool scenarios.

---

## Key Modules

- **agent.py**: `run()` for single tasks, `run_multi_tool()` for complex queries, `decompose_question()` for task breakdown
- **tool_selector.py**: LLM-based selection with fallback heuristics
- **tools.py**: Calculator (safe eval), Wikipedia (with disambiguation), DateTime (NLP parsing)
- **memory.py**: Stores recent interactions for context
- **logger.py**: Tracks performance and exports sessions

---

## Troubleshooting

**Multi-tool not triggering?**
- Use clear indicators: "and also", "and tell me"
- Or use: `multi: your question`

**Wikipedia "Multiple results"?**
- Be specific: "Python programming language" vs "Python"

**API errors?**
- Check `OPENAI_API_KEY` is set
- Verify API quota

---


---

**Built to demonstrate**: Multi-tool orchestration • Task decomposition • LLM-based reasoning • Agent architecture patterns