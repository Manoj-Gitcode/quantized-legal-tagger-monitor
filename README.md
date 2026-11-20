# Quantized Legal Tagger Monitor

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/y3n8hv8twm/badge.svg)](https://ask.thili.ai/b/y3n8hv8twm)

This project develops a tool for legal aid workers to monitor the efficiency of a legal clause tagger. You will fine-tune a GPT-2 model to classify legal clauses, then apply quantization to reduce its size and improve inference speed. The final output will be a dashboard displaying key efficiency metrics for both the original and quantized models.

## What you are building

**Goal.** You will build a monitor for legal aid workers to evaluate the size and inference speed of a quantized legal clause tagger, ensuring efficient document processing on local systems.

**Done when.** You will have successfully fine-tuned a GPT-2 model for legal clause classification, applied dynamic and static post-training quantization, and measured its impact on model size and simulated performance. The project is complete when a dashboard-like output presents these efficiency metrics.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare a subset of the LEDGAR dataset for legal clause tagging.**  
      Hint: Prepare the dataset for fine-tuning, similar to the data preparation in the 'Fine-Tune GPT-2' drill.
- [ ] **2. Fine-tune a GPT-2 model to tag legal clauses using the classification as generation approach.**  
      Hint: Apply the 'classification as generation' fine-tuning technique from the 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drill.
- [ ] **3. Evaluate the tagging accuracy of the fine-tuned model.**  
      Hint: Implement evaluation metrics to assess the model's accuracy, as demonstrated in the 'Fine-Tune GPT-2' drill.
- [ ] **4. Apply dynamic and static post-training quantization to the trained model.**  
      Hint: Use the post-training quantization techniques from the 'Shrink a Neural Network with Quantisation' drill.
- [ ] **5. Compare the original and quantized model sizes and measure simulated inference times.**  
      Hint: Measure model sizes and simulate inference times using methods from the 'Shrink a Neural Network with Quantisation' drill.
- [ ] **6. Present a dashboard-like output of model efficiency metrics.**  
      Hint: Design a clear output to present the efficiency metrics in a dashboard-like format.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Experiment with different quantization configurations or bit-widths if supported by the quantization tool.
- Add more detailed performance metrics or visualizations to the dashboard output.
- Analyze specific failure cases of the tagger and suggest potential improvements.
