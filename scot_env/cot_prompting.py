import os
import openai
from typing import Dict

# Initialize OpenAI client
client = openai.OpenAI()

# ----------------- CoT Prompt Builder -----------------

def create_cot_prompt(requirement: Dict) -> str:
    """
    Creates a Chain-of-Thought (CoT) prompt for step-by-step reasoning without structured SCoT format.
    """
    prompt = (
        "You are a skilled Python developer.\n"
        "Write a Python function to solve the following task.\n\n"
        f"Function signature: def {requirement['signature']}:\n"
        f"Docstring: \"\"\"{requirement['docstring']}\"\"\"\n\n"
        "Think step-by-step in natural language before writing the code.\n"
        "Follow this output format:\n"
        "### Reasoning\n"
        "Explain your thought process in plain language.\n\n"
        "### Final Code\n"
        "Provide the final, fully implemented Python function.\n"
    )
    return prompt

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
    os.makedirs("outputs_cot", exist_ok=True)
    filename = f"outputs_cot/{task_signature.replace(' ', '_').replace(':', '')}.txt"
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
        prompt = create_cot_prompt(task)
        output = call_model(prompt, model_name=model, max_tokens=512)
        print(f"\n--- {model} CoT Output for {task['signature']} ---\n")
        print(output)
        save_output(task['signature'], output)
