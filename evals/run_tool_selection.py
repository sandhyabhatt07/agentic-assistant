"""Tool-selection eval: accuracy and per-tool precision/recall on a labelled dataset.

python -m evals.run_tool_selection                # offline heuristics (no API key needed)
python -m evals.run_tool_selection --mode llm     # the configured LLM
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from agentic_assistant.config import Settings
from agentic_assistant.llm import create_llm
from agentic_assistant.selector import ToolSelector
from agentic_assistant.tools import default_tools

DATASET = Path(__file__).with_name("tool_selection.jsonl")


def load_dataset(path: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["offline", "llm"], default="offline")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--min-accuracy", type=float, default=0.0, help="exit 1 below this accuracy (for CI)")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    llm = create_llm(settings) if args.mode == "llm" else None
    if args.mode == "llm" and llm is None:
        print("--mode llm needs OPENAI_API_KEY", file=sys.stderr)
        return 2
    tools = {t.name: t for t in default_tools(settings, llm)}
    selector = ToolSelector(tools, llm)

    rows = load_dataset(args.dataset)
    true_pos: Counter[str] = Counter()
    predicted: Counter[str] = Counter()
    actual: Counter[str] = Counter()
    failures = []
    for row in rows:
        got = selector.select(row["task"]).tool
        predicted[got] += 1
        actual[row["expected"]] += 1
        if got == row["expected"]:
            true_pos[got] += 1
        else:
            failures.append((row["task"], row["expected"], got))

    accuracy = sum(true_pos.values()) / len(rows)
    print(f"Tool selection eval ({args.mode}, {len(rows)} cases): accuracy {accuracy:.1%}\n")
    print(f"{'tool':<12}{'precision':>10}{'recall':>10}{'support':>10}")
    for tool in sorted(actual):
        precision = true_pos[tool] / predicted[tool] if predicted[tool] else 0.0
        recall = true_pos[tool] / actual[tool]
        print(f"{tool:<12}{precision:>10.2f}{recall:>10.2f}{actual[tool]:>10}")
    if failures:
        print("\nMisses:")
        for task, expected, got in failures:
            print(f"  {task!r}: expected {expected}, got {got}")

    return 0 if accuracy >= args.min_accuracy else 1


if __name__ == "__main__":
    sys.exit(main())
