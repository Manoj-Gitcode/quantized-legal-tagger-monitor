"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest

from app.core import milestone_1


# Milestone 1: Prepare a subset of the LEDGAR dataset for legal clause tagging.
def test_milestone_1():
    subset = milestone_1()
    assert set(subset) == {"train", "validation", "test"}
    assert len(subset["train"]) == 600
    assert len(subset["validation"]) == 150
    assert len(subset["test"]) == 150
    example = subset["train"][0]
    assert example["text"].startswith("Clause:")
    assert example["text"].endswith(example["label"])
    assert len({row["label"] for row in subset["train"]}) == 10


# Milestone 2: Fine-tune a GPT-2 model to tag legal clauses using the classification as generation approach.
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Evaluate the tagging accuracy of the fine-tuned model.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Apply dynamic and static post-training quantization to the trained model.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Compare the original and quantized model sizes and measure simulated inference times.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")


# Milestone 6: Present a dashboard-like output of model efficiency metrics.
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")
