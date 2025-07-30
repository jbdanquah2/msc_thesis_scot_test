import os
import openai
from typing import List, Dict

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
    """
    Creates a prompt for SCoT (Structured Chain of Thought) prompting.

    Args:
        requirement (Dict): The target task with 'signature' and 'docstring'.
        examples (List[Dict]): Demonstration examples with signatures, docstrings,
                               SCoT reasoning steps, and reference code.

    Returns:
        str: A formatted prompt containing examples and the target task.
    """

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

    # Adding few-shot examples
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

    # Adding the target task
    prompt_lines.append(f"def {requirement['signature']}:")
    prompt_lines.append(f"    \"\"\"{requirement['docstring']}\"\"\"")
    prompt_lines.append("    # SCoT:")

    return "\n".join(prompt_lines)


# Call OpenAI model
def call_model(prompt: str, model_name: str = "gpt-3.5-turbo-1106", max_tokens: int = 512) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        top_p=0.95,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Set the OPENAI_API_KEY environment variable.")

    task1 = {
        "signature": "reverse_string(s: str) -> str",
        "docstring": "Return the reverse of the given string.",
    }

    task2 = {
        "signature": "matrix_multiply(A: list[list[int]], B: list[list[int]]) -> list[list[int]]",
        "docstring": "Given two matrices A and B, return their product. Assume valid dimensions for multiplication."
    }

    task3 = {
        "signature": "merge_intervals(intervals: list[list[int]]) -> list[list[int]]",
        "docstring": "Given a list of intervals where each interval is [start, end], merge all overlapping intervals and return the result."
    }

    task4 = {
        "signature": "count_vowels(s: str) -> int",
        "docstring": "Return the number of vowels (a, e, i, o, u) in the given string."
    }

    model = "gpt-3.5-turbo-1106"
    prompt = create_scot_prompt(task1, code_examples)
    output = call_model(prompt, model_name=model, max_tokens=512)

    print(f"\n--- {model} SCOT Output ---\n")
    print(output)
