import os
import openai
from typing import List, Dict

# Initialize OpenAI client
client = openai.OpenAI()

# ----------------- SCoT Few-Shot Examples -----------------

code_examples: List[Dict] = [
    {
        "signature": "factorial(n: int) -> int",
        "docstring": "Calculate and return the factorial of a non-negative integer n.",
        "scot": [
            "Input: n: integer",
            "Output: integer (factorial of n)",
            "1: set result to 1",
            "2: for i from 1 to n inclusive do",
            "3:     multiply result by i",
            "4: return result"
        ],
        "code": """\
def factorial(n: int) -> int:
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
""",
    },
    {
        "signature": "sum_list(nums: list[int]) -> int",
        "docstring": "Return the sum of all integers in the given list nums.",
        "scot": [
            "Input: nums: list of integers",
            "Output: integer (sum of all integers)",
            "1: set total to 0",
            "2: for each element x in nums do",
            "3:     add x to total",
            "4: return total"
        ],
        "code": """\
def sum_list(nums: list[int]) -> int:
    total = 0
    for x in nums:
        total += x
    return total
""",
    },
    {
        "signature": "is_palindrome(s: str) -> bool",
        "docstring": "Determine whether the given string s is a palindrome.",
        "scot": [
            "Input: s: string",
            "Output: boolean (True if palindrome, otherwise False)",
            "1: set left to 0",
            "2: set right to length of s minus 1",
            "3: while left is less than right do",
            "4:     if s[left] is not equal to s[right] then",
            "5:         return False",
            "6:     increment left by 1",
            "7:     decrement right by 1",
            "8: return True"
        ],
        "code": """\
def is_palindrome(s: str) -> bool:
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True
""",
    },
    {
        "signature": "merge_intervals(intervals: list[list[int]]) -> list[list[int]]",
        "docstring": "Given a list of intervals [start, end], merge all overlapping intervals and return the merged list.",
        "scot": [
            "Input: intervals: list of lists of integers [start, end]",
            "Output: list of merged intervals",
            "1: if intervals is empty then",
            "2:     return empty list",
            "3: sort intervals based on start time",
            "4: initialize merged list with the first interval",
            "5: for each interval from the second to the last in intervals do",
            "6:     set last_merged to the last interval in merged list",
            "7:     if current interval start <= last_merged end then",
            "8:         merge by setting last_merged end = max(last_merged end, current interval end)",
            "9:     else",
            "10:        append current interval to merged list",
            "11: return merged list"
        ],
        "code": """\
def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for current in intervals[1:]:
        last_merged = merged[-1]
        if current[0] <= last_merged[1]:
            last_merged[1] = max(last_merged[1], current[1])
        else:
            merged.append(current)

    return merged
""",
    }
]

# ----------------- Prompt Builder -----------------

def create_scot_prompt(requirement: Dict, examples: List[Dict]) -> str:
    """
    Creates a prompt for SCoT (Structured Chain of Thought) prompting.
    """
    intro = (
        "You are an expert software developer.\n"
        "For each problem, you will receive:\n"
        "- A function signature specifying the input and output types.\n"
        "- A docstring describing the required functionality.\n\n"
        "Your task:\n"
        "1. Think through the problem step-by-step and produce a Structured Chain-of-Thought (SCoT).\n"
        "2. The SCoT must explicitly use structured programming constructs:\n"
        "   - IO: specify inputs and outputs.\n"
        "   - Sequential steps: clearly outline ordered actions.\n"
        "   - Conditional branches: handle decision-making using 'if/else'.\n"
        "   - Loops: describe repetitive actions using 'for' or 'while'.\n\n"
        "Output Format:\n"
        "### SCoT Reasoning\n"
        "Write the structured reasoning here using numbered steps and programming constructs.\n\n"
        "### Final Code\n"
        "Write the final, complete Python function implementation here.\n\n"
        "Requirements:\n"
        "- The reasoning should be concise, clear, and logically ordered.\n"
        "- The final code must exactly match the given signature and docstring requirements.\n"
        "- Use descriptive variable names for readability.\n\n"
        "Let's think step by step and follow this structured format.\n"
    )

    prompt_lines = [intro]

    # Add few-shot examples
    for ex in examples:
        example_block = [
            f"def {ex['signature']}:",
            f"    \"\"\"{ex['docstring']}\"\"\"",
            "    ### SCoT Reasoning",
            *(f"    # {line}" for line in ex["scot"]),
            "    ### Final Code",
            ex["code"].strip(),
            ""
        ]
        prompt_lines.extend(example_block)

    # Add the target task
    prompt_lines.append(f"def {requirement['signature']}:")
    prompt_lines.append(f"    \"\"\"{requirement['docstring']}\"\"\"")
    prompt_lines.append("    ### SCoT Reasoning")

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
    os.makedirs("outputs_scot", exist_ok=True)
    filename = f"outputs_scot/{task_signature.replace(' ', '_').replace(':', '')}.txt"
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
        prompt = create_scot_prompt(task, code_examples)
        output = call_model(prompt, model_name=model, max_tokens=512)
        print(f"\n--- {model} SCOT Output for {task['signature']} ---\n")
        print(output)
        save_output(task['signature'], output)
