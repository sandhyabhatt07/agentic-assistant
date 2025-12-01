# logger.py
import logging
import json
from datetime import datetime
from pathlib import Path
from config import config

class AgentLogger:
    """Logging system for agent activities"""
    
    def __init__(self):
        self.setup_logging()
        self.session_start = datetime.now()
        self.tool_calls = []
        self.errors = []
    
    def setup_logging(self):
        """Configure logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # File handler
        logging.basicConfig(
            level=getattr(logging, config.LOG_LEVEL),
            format=log_format,
            handlers=[
                logging.FileHandler(config.LOG_FILE),
                logging.StreamHandler() if config.ENABLE_CONSOLE_LOG else logging.NullHandler()
            ]
        )
        self.logger = logging.getLogger("Agent")
    
    def log_tool_call(self, tool_name: str, input_data: str, output: str, duration: float, success: bool):
        """Log tool execution"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "input": input_data[:100],  # Truncate long inputs
            "output": output[:200] if success else output,
            "duration_ms": round(duration * 1000, 2),
            "success": success
        }
        self.tool_calls.append(log_entry)
        
        if success:
            self.logger.info(f"Tool '{tool_name}' executed successfully in {log_entry['duration_ms']}ms")
        else:
            self.logger.error(f" Tool '{tool_name}' failed: {output}")
    
    def log_error(self, error_type: str, message: str, context: dict = None):
        """Log errors"""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "message": message,
            "context": context or {}
        }
        self.errors.append(error_entry)
        self.logger.error(f"Error: {error_type} - {message}")
    
    def log_agent_start(self, task: str):
        """Log when agent starts processing"""
        self.logger.info(f"Agent started: '{task}'")
    
    def log_agent_complete(self, task: str, result: str, total_duration: float):
        """Log when agent completes"""
        self.logger.info(f"Agent completed in {round(total_duration * 1000, 2)}ms")
    
    def get_session_stats(self):
        """Get statistics for current session"""
        return {
            "session_duration": (datetime.now() - self.session_start).total_seconds(),
            "total_tool_calls": len(self.tool_calls),
            "successful_calls": sum(1 for call in self.tool_calls if call["success"]),
            "failed_calls": sum(1 for call in self.tool_calls if not call["success"]),
            "total_errors": len(self.errors)
        }
    
    def export_logs(self, filepath: str = "agent_session.json"):
        """Export session logs to JSON"""
        data = {
            "session_start": self.session_start.isoformat(),
            "session_end": datetime.now().isoformat(),
            "stats": self.get_session_stats(),
            "tool_calls": self.tool_calls,
            "errors": self.errors
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"[EXPORT] Logs exported to {filepath}")

# Global logger instance
agent_logger = AgentLogger()