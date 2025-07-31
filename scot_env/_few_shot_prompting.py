import os
import openai
from typing import List, Dict

# Initialize OpenAI client
client = openai.OpenAI()

# ----------------- Few-Shot Examples -----------------

few_shot_examples: List[Dict] = [
    {
        "signature": "factorial(n: int) -> int",
        "docstring": "Calculate and return the factorial of a non-negative integer n.",
        "code": """\
def factorial(n: int) -> int:
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
"""
    },
    {
        "signature": "sum_list(nums: list[int]) -> int",
        "docstring": "Return the sum of all integers in the given list nums.",
        "code": """\
def sum_list(nums: list[int]) -> int:
    total = 0
    for x in nums:
        total += x
    return total
"""
    },
    {
        "signature": "is_palindrome(s: str) -> bool",
        "docstring": "Determine whether the given string s is a palindrome.",
        "code": """\
def is_palindrome(s: str) -> bool:
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True
"""
    }
]

# ----------------- Few-Shot Prompt Builder -----------------

def create_few_shot_prompt(requirement: Dict, examples: List[Dict]) -> str:
    """
    Creates a few-shot prompt with examples but without SCoT reasoning.
    """
    intro = (
        "You are a skilled Python developer.\n"
        "Below are example problems with their solutions.\n"
        "Follow their style to solve the new task.\n\n"
    )

    prompt_lines = [intro]

    # Add few-shot examples
    for ex in examples:
        example_block = [
            f"def {ex['signature']}:",
            f"    \"\"\"{ex['docstring']}\"\"\"",
            ex["code"].strip(),
            ""
        ]
        prompt_lines.extend(example_block)

    # Add the target task
    prompt_lines.append("Now, solve this new task:\n")
    prompt_lines.append(f"def {requirement['signature']}:")
    prompt_lines.append(f"    \"\"\"{requirement['docstring']}\"\"\"")

    return "\n".join(prompt_lines)

# ----------------- Model Call -----------------

def call_model(prompt: str, model_name: str = "gpt-3.5-turbo-1106", max_tokens: int = 512) -> str:
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            top_p=0.95,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling model: {e}"

# ----------------- Save Output -----------------

def save_output(task_signature: str, output: str):
    os.makedirs("outputs_few_shot", exist_ok=True)
    filename = f"outputs_few_shot/{task_signature.replace(' ', '_').replace(':', '')}.txt"
    with open(filename, "w") as f:
        f.write(output)

# ----------------- Main -----------------

if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Set the OPENAI_API_KEY environment variable.")

    tasks = [
        {
            "signature": "reverse_string(s: str) -> str",
            "docstring": "Return the reverse of the given string.",
        },
        {
            "signature": "matrix_multiply(A: list[list[int]], B: list[list[int]]) -> list[list[int]]",
            "docstring": "Given two matrices A and B, return their product. Assume valid dimensions for multiplication.",
        },
        {
            "signature": "merge_intervals(intervals: list[list[int]]) -> list[list[int]]",
            "docstring": "Given a list of intervals where each interval is [start, end], merge all overlapping intervals and return the result.",
        },
        {
            "signature": "count_vowels(s: str) -> int",
            "docstring": "Return the number of vowels (a, e, i, o, u) in the given string.",
        }
    ]

    model = "gpt-3.5-turbo-1106"

    for task in tasks:
        prompt = create_few_shot_prompt(task, few_shot_examples)
        output = call_model(prompt, model_name=model, max_tokens=512)
        print(f"\n--- {model} Few-Shot Output for {task['signature']} ---\n")
        print(output)
        save_output(task['signature'], output)
