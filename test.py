# test_agent.py
import json
from sklearn.metrics import classification_report

# Import your GPT agent functions
from app import select_tool_via_llm  # type: ignore # basic GPT vs few-shot GPT
from app2 import select_tool_via_llm_fs # type: ignore

# Define test tasks and expected tools
test_data = [
    {"task": "12 + 7", "expected": "calculator"},
    {"task": "5 * 8", "expected": "calculator"},
    {"task": "Who is Elon Musk?", "expected": "wikipedia"},
    {"task": "Explain GPT-4", "expected": "wikipedia"},
    {"task": "What time is it?", "expected": "datetime"},
    {"task": "Current date please", "expected": "datetime"},
    {"task": "15 / 3", "expected": "calculator"},
    {"task": "Tell me about OpenAI", "expected": "wikipedia"},
    {"task": "Day of the week today", "expected": "datetime"},
    {"task": "2 ** 10", "expected": "calculator"},
]

def run_tests(selector_func, test_name="GPT Agent"):
    y_true = []
    y_pred = []

    print(f"\n=== Running tests for: {test_name} ===")
    for item in test_data:
        task = item["task"]
        expected = item["expected"]
        try:
            selection = selector_func(task)
            predicted = selection.get("tool") if isinstance(selection, dict) else "unknown"
        except Exception as e:
            predicted = "error"
            print(f"Error on task '{task}': {e}")

        y_true.append(expected)
        y_pred.append(predicted)

        print(f"Task: {task}\nPredicted: {predicted}, Expected: {expected}, Correct: {predicted == expected}\n")

    # Compute metrics
    accuracy = sum(p == t for p, t in zip(y_pred, y_true)) / len(y_true)
    print(f"Accuracy: {accuracy:.3f}\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=3))

if __name__ == "__main__":
    # Test basic GPT selector
    run_tests(select_tool_via_llm, "Basic GPT Selector")

    # Test GPT with few-shot examples
    run_tests(select_tool_via_llm_fs, "GPT with Few-Shot Prompt")
