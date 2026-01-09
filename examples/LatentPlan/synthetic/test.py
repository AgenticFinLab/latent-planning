from datasets import load_dataset

dataset = load_dataset("TIGER-Lab/TheoremQA")

count = 0
for d in dataset["test"]:
    if d["Picture"] is None:
        count += 1

print(count)
