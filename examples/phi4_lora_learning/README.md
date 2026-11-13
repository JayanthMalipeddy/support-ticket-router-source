# Phi-4-mini exercise data

| File | Records | Use |
|---|---:|---|
| `train.jsonl` | 14 | Seven O365 and seven Fileservice training examples |
| `original_checks.jsonl` | 6 | Original checks, excluded from training |
| `fresh_holdout.jsonl` | 6 | Additional synthetic holdout, excluded from training |

Training tickets 1 and 5 concern shared-mailbox permissions and are labelled O365.
The notebook embeds these files' contents; no separate data upload is needed.

Use [the Colab notebook](../../notebooks/Phi4_Mini_LoRA_Learning.ipynb) and follow
[the current instructions](../../docs/NOTEBOOK_GUIDE.md). Rebuild the notebook and
data with `python -B scripts/build_phi4_learning_notebook.py`.

The successful 12 November 2026 run improved training-example accuracy from
12/14 to 14/14. Both check sets remained at 6/6. The user repeated the training
check and again reported 14/14. Read the [latest-run pointer](../../records/latest.json)
and [results review](../../docs/RESULTS_REVIEW.md) for the saved evidence and limits.

Keep both check sets out of training. Better accuracy on the training examples
alone does not establish generalization. The notebook includes post-run precision
and reload fixes; the updated template still requires its own GPU execution.
