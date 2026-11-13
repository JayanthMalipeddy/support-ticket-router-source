# Current run records

`latest.json` points to the successful two-queue Phi-4-mini run:
`phi4-learning-20261112-181641-65125f`.

That directory contains original archive reports, data splits, prompt,
environment, provenance, training logs, adapter config, an import manifest,
and an offline verification report. Original report bytes are copied without
modification. `repeat-check.json` separately records the user's later 14/14
Colab check; it is explicitly user-reported, not locally rerun inference.

Binary artifacts are local under `artifacts/<run-id>/`: `run.zip` is the
original successful archive and `adapter/` holds its extracted adapter and
tokenizer. These binaries are excluded from Git and the source package.

`cleanup-20261112.json` lists obsolete resources removed during cleanup.
The current notebook includes subsequent fixes; an executed notebook from
the successful run was not included in the downloaded ZIP.

Verify the records and local artifacts with:

```powershell
python -B scripts/record_learning_run.py --verify
```
