# Phi-4-mini two-queue notebook

Use `notebooks/Phi4_Mini_LoRA_Learning.ipynb`. It contains the complete small
exercise and embeds the data in `examples/phi4_lora_learning/`.

1. Open Google Colab and choose **File > Upload notebook**.
2. Upload the notebook from this project.
3. Choose **Runtime > Change runtime type > T4 GPU**, if available.
4. Run notebook step 1 to install dependencies.
5. Choose **Runtime > Restart session**, then continue at step 2.
6. Run the remaining numbered steps in order. Run training step 8 once per run.
7. Download the results ZIP in step 10 before the runtime expires.

| Notebook step | Action |
|---|---|
| 1 | Install pinned libraries; restart the session afterward |
| 2 | Check GPU, choose native training precision, and create a run folder |
| 3 | Prepare 14 training examples and two separate sets of six checks |
| 4 | Download the original Microsoft model and record its revision |
| 5 | Measure the baseline, one ticket per request |
| 6 | Tokenize; supervise only the assistant answer and ending tokens |
| 7 | Attach LoRA and verify that base weights are frozen |
| 8 | Train for three epochs and save the changed adapter weights |
| 9 | Reload the saved adapter and evaluate using the same procedure |
| 10 | Download adapter, tokenizer, predictions, data, and training logs |

Keep evaluation examples out of training. Improvement on the 14 training
examples measures learning of those examples; fresh tickets are needed to
assess generalization. The earlier Ollama batch prompt used a different
setup, so the notebook records its own matched baseline.

## Check all 14 tickets in an active trained session

After step 9, run a new code cell:

```python
correct = 0
for number, row in enumerate(TRAIN, 1):
    prediction = classify(row["ticket"])
    matched = prediction == row["label"]
    correct += int(matched)
    print(f"{number}: {prediction} | Expected: {row['label']} | {'PASS' if matched else 'FAIL'}")
print(f"Final score: {correct}/{len(TRAIN)}")
```

This requires the model and adapter still to be loaded in that Colab session.
Restarting the runtime clears them. `ollama run phi4-mini` loads the separately
installed Ollama model; this notebook does not import its trained adapter there.

## Saved successful run and current template

The 12 November run achieved 14/14 after training and again in a user-reported
repeat check. Read its immutable reports under
`records/phi4-learning-20261112-181641-65125f/` and the current pointer in
`records/latest.json`.

After that run, the local template was improved to choose BF16 only with
native support, so a T4 selects FP16; use the `dtype` loading argument; and
verify adapter removal before clearing stale PEFT metadata on reload. Those
revisions have passed local checks but have not been rerun on a GPU. They do
not alter the saved run or guarantee that another run will have identical scores.
