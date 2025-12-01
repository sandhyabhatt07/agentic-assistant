# config.py
import os
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Agent configuration settings"""
    
    # API Settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = "gpt-3.5-turbo"
    
    # Agent Settings
    MAX_RETRIES = 3
    MAX_TOOL_ITERATIONS = 5
    REQUEST_TIMEOUT = 30
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = "agent.log"
    ENABLE_CONSOLE_LOG = True
    
    # Tool Settings
    CALCULATOR_TIMEOUT = 5
    WIKIPEDIA_SENTENCES = 2
    WEB_SCRAPER_TIMEOUT = 10
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        if not cls.OPENAI_API_KEY:
            print("⚠️  Warning: OPENAI_API_KEY not set. Using fallback heuristics.")
        return True

config = Config()