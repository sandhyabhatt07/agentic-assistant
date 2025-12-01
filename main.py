# main.py
#!/usr/bin/env python3
"""
AI Agent - Main CLI Interface
"""
import sys
from agent import Agent
from logger import agent_logger
from memory import conversation_memory
from config import config

def print_welcome():
    """Print welcome message"""
    print("\n" + "="*60)
    print("Agent:  AI AGENT - Interactive Assistant")
    print("="*60)
    print(f"Model: {config.OPENAI_MODEL}")
    print(f"Tools: calculator, wikipedia, datetime")
    print("\nCommands:")
    print("  • Type your question normally")
    print("  • 'stats' - Show agent statistics")
    print("  • 'memory' - Show conversation history")
    print("  • 'clear' - Clear conversation memory")
    print("  • 'export' - Export session logs")
    print("  • 'exit' or 'quit' - Exit the program")
    print("="*60 + "\n")

def print_stats(agent: Agent):
    """Print agent statistics"""
    stats = agent.get_stats()
    print("\n[STATS] Agent Statistics:")
    print(f"  Tools: {', '.join(stats['available_tools'])}")
    print(f"  Memory: {stats['memory']['total_interactions']} interactions")
    print(f"  Session: {stats['session']['total_tool_calls']} tool calls")
    print(f"  Success rate: {stats['session']['successful_calls']}/{stats['session']['total_tool_calls']}")
    print()

def print_memory():
    """Print conversation memory"""
    if not conversation_memory.history:
        print("\n[MEMORY] No conversation history yet.\n")
        return
    
    print("\n[MEMORY] Recent Conversation History:")
    for i, interaction in enumerate(conversation_memory.history[-5:], 1):
        print(f"\n  {i}. User: {interaction['user_input']}")
        print(f"     Tool: {interaction['tool']}")
        print(f"     Result: {interaction['result'][:100]}...")
    print()

def main():
    """Main CLI loop"""
    print_welcome()
    
    # Initialize agent
    agent = Agent()
    
    try:
        while True:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ["exit", "quit"]:
                print("\n Goodbye! Exporting logs...\n")
                agent_logger.export_logs()
                break
            
            elif user_input.lower() == "stats":
                print_stats(agent)
                continue
            
            elif user_input.lower() == "memory":
                print_memory()
                continue
            
            elif user_input.lower() == "clear":
                agent.reset()
                print("\n[FALLBACK] Memory cleared!\n")
                continue
            
            elif user_input.lower() == "export":
                agent_logger.export_logs()
                print("\n[EXPORT] Logs exported to agent_session.json\n")
                continue
            
            elif user_input.lower() == "help":
                print_welcome()
                continue
            
            # Process normal query
            print("Agent: Agent: ", end="", flush=True)
            result = agent.run(user_input)
            print(result + "\n")
    
    except KeyboardInterrupt:
        print("\n\n[WARNING]  Interrupted! Exporting logs...\n")
        agent_logger.export_logs()
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}\n")
        agent_logger.log_error("FATAL", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()