"""Your code. One stub per milestone; build them in order and keep each one small."""
import json
import random
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data")
SUBSET_PATH = DATA_DIR / "ledgar_subset.json"
MODEL_NAME = "gpt2"
MODEL_DIR = Path("models/gpt2-ledgar")
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
def build_batch(tokenizer, rows):
    """Turn a few examples into padded tensors. Only the label words count for the loss."""
    import torch

    eos = tokenizer.eos_token_id
    pairs = []
    for row in rows:
        prompt_ids = tokenizer(row["prompt"])["input_ids"]
        label_ids = tokenizer(" " + row["label"])["input_ids"] + [eos]
        pairs.append((prompt_ids, label_ids))

    longest = max(len(p) + len(l) for p, l in pairs)
    input_ids, attention_mask, labels = [], [], []
    for prompt_ids, label_ids in pairs:
        pad = longest - len(prompt_ids) - len(label_ids)
        input_ids.append(prompt_ids + label_ids + [eos] * pad)
        attention_mask.append([1] * (len(prompt_ids) + len(label_ids)) + [0] * pad)
        labels.append([-100] * len(prompt_ids) + label_ids + [-100] * pad)

    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
    }


def milestone_2(epochs=2, batch_size=8, lr=5e-5, max_examples=None, out_dir=MODEL_DIR):
    """Fine-tune GPT-2 to write the label after 'Label:' and save the model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    subset = load_subset() if SUBSET_PATH.exists() else milestone_1()
    rows = list(subset["train"])
    if max_examples:
        rows = rows[:max_examples]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    step = 0
    last_loss = None
    for epoch in range(epochs):
        random.Random(SEED + epoch).shuffle(rows)
        for start in range(0, len(rows), batch_size):
            batch = build_batch(tokenizer, rows[start:start + batch_size])
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            last_loss = loss.item()
            if step % 5 == 0:
                print(f"epoch {epoch + 1}/{epochs}  step {step}  loss {last_loss:.3f}", flush=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return {"model_dir": str(out_dir), "final_loss": last_loss, "examples": len(rows), "epochs": epochs}


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
