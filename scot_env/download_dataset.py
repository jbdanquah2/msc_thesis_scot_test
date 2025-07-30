from datasets import load_dataset

human_eval = load_dataset("openai_humaneval")["test"]
mbpp = load_dataset("mbpp")
# https://github.com/VHellendoorn/MultiPL-e
