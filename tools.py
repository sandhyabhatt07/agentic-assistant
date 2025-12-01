# tools.py - Enhanced with LLM-powered datetime
import wikipedia
from datetime import datetime, timedelta
import re
from openai import OpenAI
from config import config

def calculator(expression: str) -> str:
    """
    Evaluates mathematical expressions.
    Supports: +, -, *, /, **, parentheses.
    Example: calculator("5 + 3 * 2") returns "11"
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def wiki_search(query: str) -> str:
    """
    Searches Wikipedia and returns a summary.
    Use for: factual information, definitions, historical facts, biographies.
    Example: wiki_search("Albert Einstein") returns Einstein's biography summary.
    """
    try:
        summary = wikipedia.summary(query, sentences=2)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results found. Please be more specific. Options: {', '.join(e.options[:5])}"
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{query}'"
    except Exception as e:
        return f"Error: {e}"

def current_datetime(query: str) -> str:
    """
    Returns current date/time or calculates relative dates using natural language.
    Understands: tomorrow, yesterday, next week, day after tomorrow, 3 days from now, etc.
    Example: current_datetime("what's the date day after tomorrow?") returns the date.
    """
    try:
        # Use LLM to parse the date query
        client = OpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None
        
        if not client:
            # Fallback to simple keyword matching if no API key
            return _fallback_datetime_parsing(query)
        
        now = datetime.now()
        
        # Ask LLM to interpret the query and calculate the offset
        system_prompt = f"""You are a date/time interpreter. Current date and time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}

Parse the user's query and determine:
1. How many days to add/subtract from today (can be 0 for today, negative for past)
2. Whether they want just the date or date + time

Respond with JSON only:
{{"days_offset": <number>, "include_time": true/false, "interpretation": "<what you understood>"}}

Examples:
- "tomorrow" → {{"days_offset": 1, "include_time": false}}
- "yesterday" → {{"days_offset": -1, "include_time": false}}
- "day after tomorrow" → {{"days_offset": 2, "include_time": false}}
- "3 days ago" → {{"days_offset": -3, "include_time": false}}
- "next week" → {{"days_offset": 7, "include_time": false}}
- "what time is it" → {{"days_offset": 0, "include_time": true}}
- "today's date" → {{"days_offset": 0, "include_time": false}}
"""
        
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            max_tokens=100,
            temperature=0.0
        )
        
        # Parse LLM response
        import json
        result = json.loads(response.choices[0].message.content.strip())
        
        days_offset = result.get("days_offset", 0)
        include_time = result.get("include_time", False)
        
        # Calculate target date
        target_date = now + timedelta(days=days_offset)
        
        # Format the response
        return _format_datetime(target_date, include_time)
        
    except Exception as e:
        # Fallback to keyword matching if LLM fails
        return _fallback_datetime_parsing(query)

def _format_datetime(dt: datetime, include_time: bool = False) -> str:
    """Format datetime in a user-friendly way"""
    day_name = dt.strftime("%A")
    month = dt.strftime("%B")
    day = dt.day
    year = dt.year
    
    # Add ordinal suffix (1st, 2nd, 3rd, etc.)
    if 10 <= day % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
    
    date_str = f"{day_name}, {month} {day}{suffix}, {year}"
    
    if include_time:
        time_str = dt.strftime('%I:%M %p')
        return f"{date_str} at {time_str}"
    
    return date_str

def _fallback_datetime_parsing(query: str) -> str:
    """Fallback keyword-based parsing when LLM is unavailable"""
    query_lower = query.lower()
    now = datetime.now()
    
    # Simple keyword matching
    if "tomorrow" in query_lower:
        return _format_datetime(now + timedelta(days=1))
    elif "yesterday" in query_lower:
        return _format_datetime(now - timedelta(days=1))
    elif "day after" in query_lower or "overmorrow" in query_lower:
        return _format_datetime(now + timedelta(days=2))
    elif "next week" in query_lower:
        return _format_datetime(now + timedelta(weeks=1))
    elif "last week" in query_lower:
        return _format_datetime(now - timedelta(weeks=1))
    elif re.search(r'(\d+)\s*days?\s*(ago|from now|later)', query_lower):
        match = re.search(r'(\d+)\s*days?\s*(ago|from now|later)', query_lower)
        num_days = int(match.group(1))
        direction = match.group(2)
        offset = -num_days if direction == "ago" else num_days
        return _format_datetime(now + timedelta(days=offset))
    else:
        # Default: current date and time
        return _format_datetime(now, include_time=True)