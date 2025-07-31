import os
import openai
from typing import List, Dict, Callable
import traceback
import _scot_prompting
from _scot_prompting import code_examples, create_scot_prompt, call_model


# ---------------- PASS@k TESTING ---------------

def run_generated_code(code_str: str, test_func: Callable) -> bool:
    """Runs generated code and checks if it passes all test cases."""
    try:
        local_env = {}
        exec(code_str, {}, local_env)
        return test_func(local_env)
    except Exception as e:
        print("Execution Error:", e)
        traceback.print_exc()
        return False


def generate_multiple_samples(prompt: str, model: str, n: int = 5) -> List[str]:
    """Generates multiple code samples for Pass@k testing."""
    responses = []
    for _ in range(n):
        output = call_model(prompt, model_name=model, max_tokens=512)
        responses.append(output)
    return responses


# ----------------- UNIT TESTS -----------------

def test_reverse_string(env):
    func = env.get("reverse_string")
    if not func:
        return False
    return (func("abc") == "cba" and
            func("hello") == "olleh" and
            func("") == "")


def test_matrix_multiply(env):
    func = env.get("matrix_multiply")
    if not func:
        return False
    return func([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]]


def test_merge_intervals(env):
    func = env.get("merge_intervals")
    if not func:
        return False
    return func([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]


def test_count_vowels(env):
    func = env.get("count_vowels")
    if not func:
        return False
    return func("hello") == 2 and func("xyz") == 0


# ---------------- MAIN -----------------

if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Set the OPENAI_API_KEY environment variable.")

    tasks = [
        ({
             "signature": "reverse_string(s: str) -> str",
             "docstring": "Return the reverse of the given string."},
         test_reverse_string),
        ({
             "signature": "matrix_multiply(A: list[list[int]], B: list[list[int]]) -> list[list[int]]",
             "docstring": "Given two matrices A and B, return their product."},
         test_matrix_multiply),
        ({
             "signature": "merge_intervals(intervals: list[list[int]]) -> list[list[int]]",
             "docstring": "Merge overlapping intervals."},
         test_merge_intervals),
        ({
             "signature": "count_vowels(s: str) -> int",
             "docstring": "Return the number of vowels in the string."},
         test_count_vowels),
    ]

    model = "gpt-3.5-turbo-1106"

    for task, test_func in tasks:
        prompt = create_scot_prompt(task, code_examples)

        # Pass@1
        single_output = call_model(prompt, model_name=model, max_tokens=512)
        pass1 = run_generated_code(single_output, test_func)

        # Pass@5
        samples = generate_multiple_samples(prompt, model, n=5)
        pass5 = any(run_generated_code(sample, test_func) for sample in samples)

        print(f"\n--- Task: {task['signature']} ---")
        print(f"✅ Pass@1: {pass1}")
        print(f"✅ Pass@5: {pass5}")
