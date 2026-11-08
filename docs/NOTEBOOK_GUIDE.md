# Corrected Colab workflow

Use the updated `notebooks/Train_Phi4_Support_Router.ipynb` and rebuild `dist/support-ticket-router-source.zip` with `python -m scripts.package_source`. The old ZIP will not contain the new workflow helpers. Old evidence archives are excluded from the source ZIP.

## Recover the existing completed training

1. Upload the updated notebook to a fresh Colab GPU runtime.
2. Run the first cell, choose `recover`, and upload the new source ZIP.
3. In the recovery cell, upload the original `docs/phi4-router-adapter.zip` and `docs/week5-evidence.zip` together.
4. Continue in order. Installation creates an isolated Python environment. Enter the actual CSV column names when prompted. The regenerated split must match the original manifest exactly.
5. The notebook downloads the recorded original base revision and checks the real adapter tensors. It skips training, exports again under the pinned environment, verifies merged shards and prompt parity, and runs real smoke and held-out inference.
6. Preserve the full traceback if a cell fails. Run the evidence download cell even after a failure; it packages existing diagnostics without creating replacement scores. Do not upgrade Transformers or substitute a different classifier to bypass an error.
7. Download the evidence ZIP, executed notebook, adapter and merged weights. The evidence ZIP contains ticket text and should not be published without data redistribution permission.

## Start a new training run

Choose `train` instead of `recover`. The recovery-upload cell is skipped. The remaining cells prepare a new deterministic split, train using the configuration, and run the same integrity and inference checks. Training output is verified against its actual trainer state and tensor data; no accuracy, loss, duration or completion score is prefilled.

## Changes and limits

- ML code runs in fresh subprocesses using one isolated interpreter, avoiding a mixture of the Colab kernel's imported packages and newly installed versions.
- Core package versions match the recorded training versions; the exact factory commit is retained. These are configuration pins, not fabricated results. They still need installation verification in a fresh Colab runtime.
- Tokenization explicitly returns an input dictionary, and generation uses its actual input IDs and attention mask. Both models use cache-disabled generation to remove cache-version behavior from this troubleshooting run; its latency is not directly comparable with older runs.
- Evaluation exceptions now halt the CLI with a chained traceback. Saved reports with inference exceptions cannot be compared as completed results. Invalid generated text is still scored honestly as incorrect, never remapped to a valid label.
- Smoke cases are selected from the real prepared dataset, one per class. Expected labels are never passed to the model. Smoke cases are part of validation, not an independent quality test.
- The evidence includes source code, executed subprocess scripts, package versions before/after major stages, logs, datasets, configuration and tokenizer templates. The executed Colab notebook must still be downloaded separately.
- Nine software tests are distinct from model validation. Existing tests contain explicit test fixtures; neither notebook execution nor measured model reports use them.

The previous artifacts support completed adapter training, but not successful classification. The corrected workflow cannot guarantee accuracy or improvement. Only a completed real inference run can establish those results. The exact cause of the previous AttributeError was not recoverable from the old reports, which retained only its class name.

References used in the repair: [factory dependencies at the recorded commit](https://github.com/hiyouga/LLaMA-Factory/blob/702005ce5237ae10f3c572c3f7741ecfd1422fc9/pyproject.toml), [Hugging Face chat templates](https://huggingface.co/docs/transformers/v4.57.1/chat_templating).
