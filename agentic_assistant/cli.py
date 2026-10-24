"""Interactive command-line interface (also supports one-shot questions)."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable

from . import __version__
from .agent import Agent, AgentResponse
from .config import Settings
from .telemetry import configure_logging

logger = logging.getLogger(__name__)

COMMANDS = {
    "help": "Show this help",
    "stats": "Show agent statistics",
    "memory": "Show recent conversation history",
    "clear": "Clear conversation memory",
    "export": "Export session logs to JSON",
    "multi: <question>": "Force multi-tool mode",
    "exit / quit": "Exit (session logs are exported)",
}


def print_welcome(agent: Agent) -> None:
    mode = f"LLM ({agent.settings.openai_model})" if agent.llm else "offline heuristics (no OPENAI_API_KEY)"
    print("\n" + "=" * 60)
    print(f"  Agentic Assistant v{__version__}")
    print("=" * 60)
    print(f"Mode:  {mode}")
    print(f"Tools: {', '.join(agent.tools)}")
    print("\nCommands:")
    for command, description in COMMANDS.items():
        print(f"  {command:<20} {description}")
    print("=" * 60 + "\n")


def print_stats(agent: Agent) -> None:
    stats = agent.get_stats()
    session = stats["session"]
    print("\n[STATS]")
    print(f"  Tools:          {', '.join(stats['available_tools'])}")
    print(f"  LLM enabled:    {stats['llm_enabled']}")
    print(f"  Memory:         {stats['memory']['total_interactions']}/{stats['memory']['max_history']} interactions")
    print(f"  Tool calls:     {session['successful_calls']}/{session['total_tool_calls']} succeeded")
    print(f"  Avg latency:    {session['avg_tool_latency_ms']} ms")
    print(f"  Errors:         {session['total_errors']}\n")


def print_memory(agent: Agent) -> None:
    history = agent.memory.history
    if not history:
        print("\n[MEMORY] No conversation history yet.\n")
        return
    print("\n[MEMORY] Recent conversation history:")
    for i, item in enumerate(history[-5:], 1):
        result = item.result if len(item.result) <= 100 else item.result[:97] + "..."
        print(f"\n  {i}. You:    {item.user_input}")
        print(f"     Tool:   {item.tool}")
        print(f"     Result: {result}")
    print()


def print_response(response: AgentResponse, verbose: bool) -> None:
    if verbose or response.mode == "multi-tool":
        for i, step in enumerate(response.steps, 1):
            status = "ok" if step.success else "failed"
            print(f"  [{i}] {step.description} -> {step.tool}({step.tool_input!r}) [{status}]")
        if response.steps:
            print()
    print(f"Agent: {response.answer}\n")


def clear_memory(agent: Agent) -> None:
    agent.reset()
    print("\n[MEMORY] Memory cleared.\n")


def export_session(agent: Agent) -> None:
    try:
        path = agent.session.export(agent.settings.session_export_file)
        print(f"\n[EXPORT] Session logs exported to {path}\n")
    except OSError as exc:
        print(f"\n[ERROR] Could not export session logs: {exc}\n")


def handle_command(agent: Agent, command: str) -> bool:
    """Run a built-in command. Returns True when ``command`` was one."""
    handlers: dict[str, Callable[[], None]] = {
        "help": lambda: print_welcome(agent),
        "stats": lambda: print_stats(agent),
        "memory": lambda: print_memory(agent),
        "clear": lambda: clear_memory(agent),
        "export": lambda: export_session(agent),
    }
    handler = handlers.get(command.lower())
    if handler is None:
        return False
    handler()
    return True


def repl(agent: Agent, verbose: bool = False) -> int:
    print_welcome(agent)
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        if handle_command(agent, user_input):
            continue

        try:
            if user_input.lower().startswith("multi:"):
                response = agent.answer_multi_tool(user_input[len("multi:") :].strip())
            else:
                response = agent.ask(user_input)
            print_response(response, verbose)
        except KeyboardInterrupt:
            print("\n[INTERRUPTED]\n")
        except Exception as exc:  # keep the session alive; details go to the log file
            logger.exception("Unhandled error while answering %r", user_input)
            agent.session.log_error("UNHANDLED", str(exc), {"input": user_input})
            print(f"\n[ERROR] Something went wrong: {exc}\n")

    print("Goodbye!")
    export_session(agent)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic-assistant",
        description="A multi-tool AI assistant (calculator, Wikipedia, date/time).",
    )
    parser.add_argument("question", nargs="*", help="ask one question and exit (omit for interactive mode)")
    parser.add_argument("-v", "--verbose", action="store_true", help="show the tool calls behind each answer")
    parser.add_argument("--offline", action="store_true", help="never call the LLM, use rule-based fallbacks")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.from_env()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    configure_logging(settings)
    agent = Agent(settings, use_llm=not args.offline)

    if args.question:
        response = agent.ask(" ".join(args.question))
        if args.verbose:
            print_response(response, verbose=True)
        else:
            print(response.answer)
        return 0 if response.success else 1
    return repl(agent, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
