import re

def remove_emojis_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace common emojis with text alternatives
    replacements = {
        '🎯': '[SELECTED]',
        '🔄': '[FALLBACK]',
        '⚠️': '[WARNING]',
        '📄': '[EXPORT]',
        '✅': '[SUCCESS]',
        '❌': '[ERROR]',
        '🤖': 'Agent:',
        '🧑': 'You:',
        '👋': '',
        '📊': '[STATS]',
        '💭': '[MEMORY]',
    }
    
    for emoji, replacement in replacements.items():
        content = content.replace(emoji, replacement)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed {filepath}")

# Fix all files
files = ['tool_selector.py', 'agent.py', 'logger.py', 'main.py']
for file in files:
    try:
        remove_emojis_from_file(file)
    except FileNotFoundError:
        print(f"Skipped {file} (not found)")