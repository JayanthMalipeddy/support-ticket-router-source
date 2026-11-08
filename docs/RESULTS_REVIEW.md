# Colab results review — 6 November 2026

**Assessment: the supplied artifacts support a completed training run, but the evaluation failed. Model routing quality is not yet established. Do not retrain solely because these reports show 0% accuracy.**

This is an offline review of `week5-evidence.zip` and `phi4-router-adapter.zip`. No model inference was rerun locally. Original archives were left unchanged.

## Second review — 6 November 2026

Both archive SHA-256 hashes are unchanged from the first review. No new executed notebook, source snapshot or corrected evaluation reports were found in `docs` or inside the evidence archive. The conclusions below therefore still apply.

Additional checkpoint verification strengthens the training evidence: all 256 adapter tensors differ between checkpoint 59 and the final adapter, and all 256 differ between checkpoint 118 and the final adapter. The final adapter exactly matches checkpoint 177. All inspected checkpoint tensors are finite; the final adapter's 128 LoRA B tensors contain 7,077,888 nonzero values. This supports actual parameter updates across training, but does not establish classification quality or that the adapter was correctly incorporated into the merged model.

The second review again verified all 118 unique validation IDs, case ordering, reconstructed validation fingerprint, metrics and comparison arithmetic. All 236 evaluation calls across both models are recorded as `AttributeError`; none yielded a valid label. The five saved smoke outputs also contain no valid queue labels. There is no evidence here that the model needs retraining, nor enough diagnostic detail to identify the exact inference failure. The full direct-prediction traceback remains the next required evidence.

## Training evidence

| Item | Recorded result |
|---|---|
| Model | microsoft/Phi-4-mini-instruct |
| GPU | Tesla T4, 15,637,086,208 bytes VRAM |
| Split | 467 training / 118 validation tickets |
| Epochs / optimizer steps | 3 / 177 of 177 |
| Training runtime | 619.3553 seconds (10 minutes 19 seconds) |
| Mean training loss | 0.231445 |
| First / last logged window loss | 0.9701 / 0.0750 |
| Adapter | LoRA rank 8, alpha 16; 256 tensors, 11,534,336 parameters |
| Preflight | Reports prompt parity and successful CUDA 4-bit operation |
| Maximum training sequence | 171 tokens |

Both ZIP archives passed CRC checks. Final trainer state matches across archives. The final adapter's safetensors payload contains finite, nonzero float32 values throughout. These checks support the presence of a structurally readable trained adapter, not its routing accuracy. Loss decreases overall with fluctuations; there is no held-out loss curve from which to infer generalization or overfitting.

The actual training time is much shorter than the earlier rough planning estimate. Use this recorded 10m19s figure for this run only; it excludes setup, checkpoint downloads, merge and evaluation.

## Evaluation findings

| Measure | Baseline | Fine-tuned |
|---|---:|---:|
| Evaluation cases | 118 | 118 |
| Cases with AttributeError | 118 | 118 |
| Successful prediction calls | 0 | 0 |
| Recorded accuracy | 0% | 0% |
| Recorded macro-F1 | 0 | 0 |
| Median inference latency | null | null |

Every case has `label: null`, `latency_ms: null`, and `error: "AttributeError"`. The scorer counts those failures as incorrect, so the zero metrics are arithmetically consistent but **not a valid comparison of model classification ability**. The zero improvement in `comparison.json` does not establish that fine-tuning had no effect. There is no usable speed comparison.

I recomputed the metrics and comparison from the saved predictions and checked their equality after JSON normalization. I reconstructed all 118 validation rows from the Week 4 exchange cases; their fingerprint matches the manifest and both reports. Case IDs, expected labels and order agree across both reports and the exchange cases. The training rows/raw CSV were not supplied, so train/validation leakage and source content were not independently audited.

All five saved smoke outputs are prose responses/refusals instead of allowed queue labels: 0/5 exact matches. Their keys (`raw`, `ms`, `tokens`) and unrestricted `label` values differ from the repository classifier's contract (`raw_output`, `latency_ms`, strict label or null). Thus the smoke result was produced through a changed code path; it cannot verify the checked-in inference path. The executed notebook and run-time source were not supplied, so the exact change is unknown.

## Likely investigation direction, not a confirmed root cause

The recorded environment lists Transformers **4.57.6**. The merged model's `config.json` and `generation_config.json` record **5.22.1**, outside this project's declared `<5` inference dependency. The adapter tokenizer identifies `GPT2Tokenizer`, while the merged tokenizer identifies `TokenizersBackend`. These artifacts indicate environment/serialization changes between stages, or artifacts from different environments. The installation-time package list does not establish which version was active during evaluation.

The evaluator retains only exception class names, so it discarded the traceback required to locate the AttributeError. It is not possible to tell from these reports whether it came from tokenizer output handling, generation/cache code, or a changed classifier implementation. Do not change prompts or retrain as a substitute for diagnosing the exception.

## Recovery sequence

1. Preserve these archives and the trained adapter. Download the executed Colab notebook and the actual `ticket_router/inference.py` used in Colab, including any edits made during troubleshooting.
2. In the existing GPU runtime, call `HFClassifier.predict()` directly on one authored ticket. This bypasses evaluation's exception handling and exposes the full traceback. Record current package versions and module path in the same cell (example below).
3. Repair the identified inference error and restore the strict classification prompt/output contract. Resolve the environment drift before re-exporting or evaluating. If a fresh export is necessary, reuse the saved adapter and exact original base revision; do not repeat training unless additional evidence justifies it.
4. Check direct prediction with the base model and with the adapter/merged model under one compatible environment. Inspect raw outputs and verify that the intended model and prompt are loaded. Classification errors should remain errors, not be remapped into plausible labels.
5. Rerun baseline and candidate on the same 118 held-out cases, using new output filenames such as `baseline-rerun.json`, `tuned-rerun.json` and `comparison-rerun.json`. Keep these failed reports for provenance.
6. Export fresh environment details after merge/evaluation, the executed notebook, exact source, new smoke results and reports. Preserve the merged weights or a reproducible adapter-plus-base recovery path. Only then assess accuracy, macro-F1, per-class weaknesses and submission claims.

Run this in Colab after ensuring no other model is occupying GPU memory. If it fails, preserve the full traceback:

```python
import importlib.metadata as metadata
import ticket_router.inference as inference

print("Classifier source:", inference.__file__)
for package in ("torch", "transformers", "peft", "accelerate", "bitsandbytes"):
    print(package, metadata.version(package))

classifier = inference.HFClassifier("artifacts/phi4-router-merged")
try:
    print(classifier.predict("Outlook messages are stuck in my Outbox."))
finally:
    del classifier
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
```

The supplied archives contain adapter weights and checkpoints but no merged model weight shards. The evidence ZIP intentionally omits weight files and also omits `.jinja` templates; its merged tokenizer files alone are not a complete deployable model. Absence from that ZIP does not prove the original merged directory lacked its template or weights.

## Provenance

- Base revision: `cfbefacb99257ffa30c83adab238a50856ac3083`
- LLaMA Factory commit: `702005ce5237ae10f3c572c3f7741ecfd1422fc9`
- Validation fingerprint: `90269d08dfb57afdeb9106931f9686620e8562dc8f4f0f2cd59f4abddc78f037`
- Evidence ZIP SHA-256: `55cfc42b9c688dd3fc84eba55d9e3cf14a80e09d7d0ca407cc9f31dc3fb5e2b6`
- Adapter ZIP SHA-256: `bdb896da5e3ac12d57a5d00b86eb36112505059ef59d5deb90f633d07fc0db19`

Week 3 and Week 4 integration boundaries remain unchanged. Nothing was published or submitted.
