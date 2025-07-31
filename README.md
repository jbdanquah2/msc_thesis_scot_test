

# 🧠 Structured Chain-of-Thought (SCoT) Code Generation

This repository provides scripts to replicate **Structured Chain-of-Thought (SCoT)** prompting for code generation, along with baseline prompting techniques (**Zero-shot**, **Few-shot**, and **Chain-of-Thought (CoT)**). These scripts can be used to reproduce SCoT experiments or compare prompting methods.

---

## 📂 Repository Contents

* `scorm_prompting.py` → **SCoT prompting** (structured reasoning with IO, branches, loops).
* `zero_shot_prompting.py` → Baseline Zero-shot prompting.
* `few_shot_prompting.py` → Baseline Few-shot prompting.
* `cot_prompting.py` → Baseline Chain-of-Thought prompting.
* `outputs/` → SCoT generated code solutions.
* `outputs_zero_shot/`, `outputs_few_shot/`, `outputs_cot/` → Baseline results.

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
API key can be found in the openAIK.txt file
```bash
export OPENAI_API_KEY="your_api_key_here"
```

## *(On Windows PowerShell: `setx OPENAI_API_KEY "your_api_key_here"`)*

## ▶️ Running Scripts

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

* Generated solutions can be tested against datasets like **HumanEval** or **MBPP**.
* To evaluate:

  1. Generate solutions using any script.
  2. Run test cases for each solution.
  3. Calculate unbiased Pass\@1, Pass\@3, Pass\@5:

     $$
     Pass@k = 1 - \frac{\binom{n - c}{k}}{\binom{n}{k}}
     $$

     Where `n` = total generations, `c` = correct solutions.

---

## 🧭 Choosing Which Script to Run

| Script                   | Prompting Type                         | When to Use                                                      | Output Folder        |
| ------------------------ | -------------------------------------- | ---------------------------------------------------------------- | -------------------- |
| `scorm_prompting.py`     | **SCoT** (Structured Chain-of-Thought) | Main methodology. Structured reasoning with IO, branches, loops. | `outputs/`           |
| `zero_shot_prompting.py` | Zero-shot                              | Baseline: Directly asks for solution with no examples.           | `outputs_zero_shot/` |
| `few_shot_prompting.py`  | Few-shot                               | Baseline: Uses code examples to guide the solution.              | `outputs_few_shot/`  |
| `cot_prompting.py`       | Chain-of-Thought                       | Baseline: Natural language step-by-step reasoning.               | `outputs_cot/`       |

---

## 📊 Prompting Flowchart

![Prompting Scripts Overview](Prompting_Scripts_Overview.png)

---

## 📄 Notes

* To reproduce **only SCoT results**, run `scorm_prompting.py`.
* Baseline scripts (Zero-shot, Few-shot, CoT) are **optional**, only needed for performance comparisons.
* To fully replicate the research:

  * Use HumanEval and MBPP datasets.
  * Generate **20 solutions per task**.
  * Compute unbiased **Pass\@k** scores.
  * Optionally, perform robustness and ablation studies.


