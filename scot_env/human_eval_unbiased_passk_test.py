import os
from datasets import load_dataset
from typing import List, Dict, Callable
import math
import subprocess
import tempfile
from _scot_prompting import code_examples, create_scot_prompt, call_model

# ----------------- Helper Functions -----------------

def run_generated_code(code_str: str, test_code: str) -> bool:
    """Runs generated code and checks if it passes test cases."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        tmp.write(code_str + "\n" + test_code)
        tmp_path = tmp.name
    try:
        subprocess.check_output(["python", tmp_path], stderr=subprocess.STDOUT, timeout=10)
        return True
    except subprocess.CalledProcessError:
        return False
    finally:
        os.remove(tmp_path)


def generate_multiple_samples(prompt: str, model: str, n: int = 20) -> List[str]:
    responses = []
    for _ in range(n):
        output = call_model(prompt, model_name=model, max_tokens=512)
        responses.append(output)
    return responses


def unbiased_pass_k(n: int, c: int, k: int) -> float:
    if n < k:
        return 0.0
    return 1 - math.comb(n - c, k) / math.comb(n, k)


def extract_code(output: str) -> str:
    return "\n".join(line for line in output.splitlines() if not line.strip().startswith("#"))

# ----------------- Dataset Formatting -----------------

def format_humaneval_task(row: Dict) -> Dict:
    lines = row["prompt"].splitlines()
    sig = next((line[4:].strip() for line in lines if line.strip().startswith("def ")), None)
    docstring = next((line.strip().strip('"') for line in lines if '"""' in line), "")
    return {"signature": sig, "docstring": docstring, "test_code": row["test"]}

# ----------------- Main -----------------
if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Set the OPENAI_API_KEY environment variable.")

    model = "gpt-3.5-turbo-1106" # set the desired model her

    dataset = load_dataset("openai_humaneval") # load the HumanEval dataset
    tasks = [format_humaneval_task(row) for row in dataset["test"]] # format the dataset for SCoT

    # reducing the number of tasks for faster testing
    tasks = tasks[:20]  # for faster testing, comment this line out for full evaluation

    total_tasks = len(tasks)
    pass1_count, pass3_scores, pass5_scores = 0, [], []

    for i, task in enumerate(tasks):
        print(f"Task {i+1}/{total_tasks}: {task['signature']}")
        prompt = create_scot_prompt(task, code_examples)

        # Generate samples
        samples = generate_multiple_samples(prompt, model, n=2) # reduced to 2 for faster testing, should be increased to 20 for full evaluation
        successes = sum(run_generated_code(extract_code(sample), task["test_code"]) for sample in samples)

        # Pass@1

        pass1 = successes > 0 and run_generated_code(extract_code(samples[0]), task["test_code"])
        if pass1:
            pass1_count += 1

        # Store unbiased Pass@3 and Pass@5 for later aggregation
        pass3_scores.append(unbiased_pass_k(20, successes, 3))
        pass5_scores.append(unbiased_pass_k(20, successes, 5))

    print(f"\nPass@1: {pass1_count}/{total_tasks} = {pass1_count / total_tasks:.3f}")
    print(f"Average Pass@3: {sum(pass3_scores) / total_tasks:.3f}")
    print(f"Average Pass@5: {sum(pass5_scores) / total_tasks:.3f}")
