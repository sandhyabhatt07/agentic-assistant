import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools import calculator, wiki_search



def test_calculator_works():
    result = calculator("5 + 3")
    assert result == "8"

def test_calculator_multiplication():
    result = calculator("4 * 5")
    assert result == "20"

def test_wiki_returns_something():
    result = wiki_search("Python programming")
    assert len(result) > 0