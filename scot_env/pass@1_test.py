import os
import openai
from typing import List, Dict, Callable
import traceback

# Initialize OpenAI client
client = openai.OpenAI()

# Demonstration examples for SCOT
code_examples: List[Dict] = [
    {
        "signature": "factorial(n: int) -> int",
        "docstring": "Return the factorial of a non-negative integer n.",
        "scot": [
            "IO: input n: int, output: int",
            "If n is 0 or 1:",
            "    return 1",
            "Else:",
            "    return n * factorial(n - 1)",
        ],
        "code": """\ndef factorial(n: int) -> int:\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n - 1)\n""",
    },
    {
        "signature": "sum_list(nums: list[int]) -> int",
        "docstring": "Return the sum of all integers in the list nums.",
        "scot": [
            "IO: input nums: list of int, output: int",
            "Initialize total to 0",
            "Loop through each element x in nums:",
            "    add x to total",
            "Return total",
        ],
        "code": """\ndef sum_list(nums: list[int]) -> int:\n    total = 0\n    for x in nums:\n        total += x\n    return total\n""",
    },
    {
        "signature": "is_palindrome(s: str) -> bool",
        "docstring": "Check whether the given string is a palindrome.",
        "scot": [
            "IO: input s: str, output: bool",
            "Set left index to 0 and right index to len(s) - 1",
            "While left is less than right:",
            "    If s[left] != s[right]: return False",
            "    Increment left and decrement right",
            "Return True",
        ],
        "code": """\ndef is_palindrome(s: str) -> bool:\n    left, right = 0, len(s) - 1\n    while left < right:\n        if s[left] != s[right]:\n            return False\n        left += 1\n        right -= 1\n    return True\n""",
    },
]


def create_scot_prompt(requirement: Dict, examples: List[Dict]) -> str:
    intro = (
        "You are an expert developer.\n"
        "For each problem, you will be given:\n"
        "- A function signature\n"
        "- A docstring describing the required functionality\n\n"
        "Your task:\n"
        "1. Produce a structured chain of thought (SCoT) using programming structures:\n"
        "   - IO specification\n"
        "   - Sequential steps\n"
        "   - Conditional branches\n"
        "   - Loops\n"
        "2. After reasoning, output only the final code.\n\n"
        "3. The SCoT should be clear and concise, guiding the implementation.\n"
        "4. The final code should be a complete function that adheres to the signature and docstring.\n\n"
        "4. Remember to use meaningful variable names.\n"
        "Let's think step by step.\n"
    )

    prompt_lines = [intro]
    for ex in examples:
        example_block = [
            f"def {ex['signature']}:",
            f"    \"\"\"{ex['docstring']}\"\"\"",
            "    # SCoT:",
            *(f"    # {line}" for line in ex["scot"]),
            ex["code"].strip(),
            ""
        ]
        prompt_lines.extend(example_block)

    prompt_lines.append(f"def {requirement['signature']}:")
    prompt_lines.append(f"    \"\"\"{requirement['docstring']}\"\"\"")
    prompt_lines.append("    # SCoT:")
    return "\n".join(prompt_lines)


def call_model(prompt: str, model_name: str = "gpt-3.5-turbo-1106", max_tokens: int = 512) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        top_p=0.95,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


# ---------------- PASS@1 TESTING ----------------

def run_generated_code(code_str: str, test_func: Callable) -> bool:
    """
    Runs generated code and checks if it passes all test cases.
    """
    try:
        local_env = {}
        exec(code_str, {}, local_env)
        return test_func(local_env)
    except Exception as e:
        print("Execution Error:", e)
        traceback.print_exc()
        return False


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
    return func([[1,2],[3,4]], [[5,6],[7,8]]) == [[19,22],[43,50]]


def test_merge_intervals(env):
    func = env.get("merge_intervals")
    if not func:
        return False
    return func([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]


def test_count_vowels(env):
    func = env.get("count_vowels")
    if not func:
        return False
    return func("hello") == 2 and func("xyz") == 0


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Set the OPENAI_API_KEY environment variable.")

    tasks = [
        ({"signature": "reverse_string(s: str) -> str", "docstring": "Return the reverse of the given string."}, test_reverse_string),
        ({"signature": "matrix_multiply(A: list[list[int]], B: list[list[int]]) -> list[list[int]]", "docstring": "Given two matrices A and B, return their product."}, test_matrix_multiply),
        ({"signature": "merge_intervals(intervals: list[list[int]]) -> list[list[int]]", "docstring": "Merge overlapping intervals."}, test_merge_intervals),
        ({"signature": "count_vowels(s: str) -> int", "docstring": "Return the number of vowels in the string."}, test_count_vowels),
    ]

    model = "gpt-3.5-turbo-1106"

    for task, test_func in tasks:
        prompt = create_scot_prompt(task, code_examples)
        output = call_model(prompt, model_name=model, max_tokens=512)

        print(f"\n--- Task: {task['signature']} ---")
        print(output)

        passed = run_generated_code(output, test_func)
        print(f"✅ Pass@1: {passed}")
