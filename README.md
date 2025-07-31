
# 🧠 Structured Chain-of-Thought (SCoT) Code Generation

This repository provides Python scripts to replicate **Structured Chain-of-Thought (SCoT)** prompting for code generation, along with baseline prompting techniques (**Zero-shot**, **Few-shot**, and **Chain-of-Thought (CoT)**). It also includes scripts for evaluating performance on benchmarks using **Pass\@k metrics**.

These scripts can be used to reproduce the SCoT methodology or compare prompting methods against baselines.

---

## 📂 Repository Contents

### Prompting Scripts

* `_scot_prompting.py` → **SCoT prompting** (structured reasoning with IO, branches, loops).
* `_zero_shot_prompting.py` → Baseline Zero-shot prompting.
* `_few_shot_prompting.py` → Baseline Few-shot prompting.
* `_cot_prompting.py` → Baseline Chain-of-Thought prompting.

### Evaluation Scripts

* `human_eval_unbiased_passk_test.py` → Run unbiased Pass\@k evaluation on the **HumanEval** dataset.
* `mbpp_unbiased_passk_test.py` → Run unbiased Pass\@k evaluation on the **MBPP** dataset.
* `passk_test.py` → General Pass\@k testing for custom tasks.

### Output Folders

* `outputs/` → Generated solutions for SCoT.
* `outputs_zero_shot/` → Generated solutions for Zero-shot.
* `outputs_few_shot/` → Generated solutions for Few-shot.
* `outputs_cot/` → Generated solutions for Chain-of-Thought.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-link>
cd <your-repo-name>
```

### 2. Install dependencies

```bash
pip install openai datasets
```

### 3. Set your OpenAI API key

API key can be found in the `openAIK.txt` file.

```bash
export OPENAI_API_KEY="your_api_key_here"
```

## *(On Windows PowerShell: `setx OPENAI_API_KEY "your_api_key_here"`)*

## ▶️ Running Prompting Scripts

### **SCoT Prompting**

Structured reasoning before code generation:

```bash
python scorm_prompting.py
```

* Uses structured reasoning.
* Saves results to `outputs/`.

### **Zero-shot Prompting**

No examples, direct solution:

```bash
python zero_shot_prompting.py
```

* Saves results to `outputs_zero_shot/`.

### **Few-shot Prompting**

Uses examples to guide code style:

```bash
python few_shot_prompting.py
```

* Saves results to `outputs_few_shot/`.

### **Chain-of-Thought (CoT) Prompting**

Plain step-by-step reasoning:

```bash
python cot_prompting.py
```

* Saves results to `outputs_cot/`.

---

## 🧪 Pass\@k Evaluation

Scripts for evaluating generated solutions against benchmark datasets:

### HumanEval

```bash
python human_eval_unbiased_passk_test.py
```

### MBPP

```bash
python mbpp_unbiased_passk_test.py
```

### Custom Tasks

```bash
python passk_test.py
```

### Formula:

$$
Pass@k = 1 - \frac{\binom{n - c}{k}}{\binom{n}{k}}
$$

Where:

* `n` = total generations
* `c` = correct solutions

---

## 🧭 Choosing Which Script to Run

| Script                              | Prompting Type         | When to Use                                   | Output Folder        |
| ----------------------------------- | ---------------------- | --------------------------------------------- | -------------------- |
| `_scot_prompting.py`                | **SCoT**               | Structured reasoning with IO, branches, loops | `outputs/`           |
| `_zero_shot_prompting.py`           | Zero-shot              | Direct solution with no examples              | `outputs_zero_shot/` |
| `_few_shot_prompting.py`            | Few-shot               | Uses code examples to guide solution style    | `outputs_few_shot/`  |
| `_cot_prompting.py`                 | Chain-of-Thought       | Natural language step-by-step reasoning       | `outputs_cot/`       |
| `human_eval_unbiased_passk_test.py` | Evaluation (HumanEval) | Pass\@k testing on HumanEval benchmark        | N/A                  |
| `mbpp_unbiased_passk_test.py`       | Evaluation (MBPP)      | Pass\@k testing on MBPP benchmark             | N/A                  |
| `passk_test.py`                     | Evaluation (Custom)    | Pass\@k testing for your custom tasks         | N/A                  |

---

## 📄 Notes

* To reproduce **only SCoT results**, run `scot_prompting.py`.
* Baseline scripts (Zero-shot, Few-shot, CoT) are **optional**, used for performance comparison.
* To fully replicate the research:

  * Use HumanEval and MBPP datasets.
  * Generate **20 solutions per task**.
  * Compute unbiased **Pass\@1, Pass\@3, and Pass\@5** scores.
  * Optionally, perform robustness and ablation studies.


