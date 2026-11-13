# Verification of the current successful run

Current run: `phi4-learning-20261112-181641-65125f`, completed in the user's
Google Colab GPU session on 12 November 2026.

## Executed evidence

The saved `before.json`, `after.json`, `comparison.json`, and `training.json`
record 21 optimizer steps, training accuracy from 12/14 to 14/14, and unchanged
6/6 scores on each of two separate check sets. The user reported another
14/14 prediction run in the same active trained-model session.

## Local offline checks

`python -B scripts/record_learning_run.py --verify` verifies:

- The original ZIP's expected SHA-256 and CRC checksums.
- All extracted-file sizes and SHA-256 hashes against the saved manifest.
- Exact agreement of saved data with the current exercise data.
- Seven training examples per label and three per label in each check set.
- No exact text duplicates across training and check sets.
- Before/after case identity, response grading, score arithmetic, and comparison.
- 128 finite FP32 adapter tensors with 3,670,016 total parameters.
- 64 nonzero LoRA B matrices and recorded adapter-weight changes.
- Model identity, base revision, rank, alpha, target modules, and finite loss.

Earlier local notebook checks compiled all ten code cells and used Microsoft's
actual tokenizer to verify the training/inference chat boundary, answer-only
loss, and padding masks for all 14 examples. Grading checks confirmed that
explanatory prose scores as invalid and inference exceptions propagate.

Run `python -B -m unittest discover -s tests -v` for the retained API and data
utility fixture tests. Those fixtures do not measure trained model accuracy.
Test-generated data is cleaned after each test.

## Limits and current notebook revision

Local verification did not rerun GPU training or inference. The executed
Colab notebook was not supplied. The current local notebook includes later
precision and adapter-reload checks; that exact revised template has not
been executed on a GPU. Original environment and prediction records are
preserved without rewriting them to match the updated template.

No full merged model, Ollama import, local API deployment of this adapter,
latency improvement, or production-quality evaluation is verified.
