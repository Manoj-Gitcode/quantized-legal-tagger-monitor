"""Your code. One stub per milestone; build them in order and keep each one small."""
import json
import random
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data")
SUBSET_PATH = DATA_DIR / "ledgar_subset.json"
SEED = 42


# --- Milestone 1: Prepare a subset of the LEDGAR dataset for legal clause tagging.
def format_example(clause, label_name):
    """Turn one clause into the text format GPT-2 will learn from."""
    clause = " ".join(clause.split())  # remove line breaks and extra spaces
    prompt = f"Clause: {clause}\nLabel:"
    return {"prompt": prompt, "label": label_name, "text": f"{prompt} {label_name}"}


def milestone_1(top_labels=10, train_per_label=60, eval_per_label=15, max_chars=600):
    """Download LEDGAR, keep a small balanced subset, save it, and return it."""
    from datasets import load_dataset

    ds = load_dataset("coastalcph/lex_glue", "ledgar")
    label_names = ds["train"].features["label"].names

    # the most common labels in the training data
    counts = Counter(ds["train"]["label"])
    keep = {label for label, _ in counts.most_common(top_labels)}

    sizes = {"train": train_per_label, "validation": eval_per_label, "test": eval_per_label}
    subset = {}
    for split, per_label in sizes.items():
        taken = Counter()
        rows = []
        for row in ds[split].shuffle(seed=SEED):
            label = row["label"]
            if label in keep and taken[label] < per_label:
                taken[label] += 1
                rows.append(format_example(row["text"][:max_chars], label_names[label]))
            if len(rows) == per_label * len(keep):
                break
        random.Random(SEED).shuffle(rows)
        subset[split] = rows

    DATA_DIR.mkdir(exist_ok=True)
    SUBSET_PATH.write_text(json.dumps(subset, indent=2))
    return subset


def load_subset():
    """Read the saved subset (used by the later milestones)."""
    return json.loads(SUBSET_PATH.read_text())


# --- Milestone 2: Fine-tune a GPT-2 model to tag legal clauses using the classification as generation approach.
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Evaluate the tagging accuracy of the fine-tuned model.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Apply dynamic and static post-training quantization to the trained model.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Compare the original and quantized model sizes and measure simulated inference times.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")


# --- Milestone 6: Present a dashboard-like output of model efficiency metrics.
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")
