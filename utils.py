# utils.py
import json
import re
from typing import Dict, Any

def safe_parse_json(text: str) -> Dict[str, Any]:
    """Try to parse JSON from text, fallback to empty dict on error."""
    try:
        # Remove markdown code blocks if present
        text = re.sub(r'```json\s*|\s*```', '', text)
        
        # Find JSON object
        start = text.index("{")
        end = text.rindex("}") + 1
        json_str = text[start:end]
        
        return json.loads(json_str)
    except Exception as e:
        return {}

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

def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to max_length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 1:
        return f"{round(seconds * 1000)}ms"
    elif seconds < 60:
        return f"{round(seconds, 2)}s"
    else:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"