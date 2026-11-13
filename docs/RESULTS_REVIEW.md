# Successful Phi-4-mini run: 12 November 2026

Run: `phi4-learning-20261112-181641-65125f`.
The downloaded ZIP supports completed QLoRA training, a saved adapter, and
successful prediction calls before and after training.

| Dataset | Before | After |
|---|---:|---:|
| 14 training tickets | 12/14 (85.7%) | 14/14 (100%) |
| Original six checks, outside training | 6/6 | 6/6 |
| Fresh six synthetic held-out tickets | 6/6 | 6/6 |

In this notebook baseline, training tickets **5 and 13** were incorrect. Both
were correct after training. The user subsequently ran all 14 through the
loaded trained model again and reported 14/14; that repeat is recorded as
user-reported evidence, separate from the original archive predictions.

The 14.3 percentage-point change is training-example accuracy. Both held-out
sets were already perfect, so this experiment measured no held-out accuracy
gain. The small synthetic sets cannot establish production quality. No latency
improvement, cost reduction, calibrated confidence, merged-model validation,
or successful Ollama deployment is claimed.

## Training evidence

- Base: `microsoft/Phi-4-mini-instruct` at `cfbefacb99257ffa30c83adab238a50856ac3083`.
- GPU: Tesla T4; Python 3.13.16; PyTorch 2.14.1+cu130; CUDA 13.0.
- Transformers 4.57.6; PEFT 0.18.1; Accelerate 1.11.0; bitsandbytes 0.50.2.
- Recorded compute dtype: BF16, as saved in the original run.
- NF4 four-bit base, double quantization; rank 8; alpha 16; dropout 0.05.
- Attention targets: `qkv_proj`, `o_proj`.
- Three epochs; batch 1; gradient accumulation 2; learning rate 0.0002.
- 21 optimizer steps; 3,670,016 trainable adapter parameters.
- Mean training loss: 0.07643016459908158.
- Recorded sum of absolute adapter-weight changes: 2737.7521257400513.

Local archive inspection verified CRC integrity, report arithmetic, case/data
identity, exact-duplicate split separation, 128 finite FP32 adapter tensors,
and 64 nonzero LoRA B matrices. This inspection did not rerun model inference.

## Warnings and template corrections

The public Microsoft model downloaded without an HF token. The `torch_dtype`
deprecation did not stop loading. Step 9 completed with an existing
`peft_config` warning after unloading and reattaching the saved adapter.
PEFT's unload path removes adapter layers but can retain that metadata.

The current local notebook checks that adapter layers and parameters were
removed before clearing the stale metadata, and verifies one reloaded adapter.
It also uses `dtype` and excludes emulated BF16 when selecting precision for
future runs. The saved run's BF16 environment record remains unchanged.
These post-run code changes are locally checked and await a new GPU run.

## Provenance

Original archive SHA-256:
`e0bb758433be4aa9318899696bd15d26fbacde40d8ccd50e2654671165d8c3b0`.

The original ZIP and extracted adapter are under
`artifacts/phi4-learning-20261112-181641-65125f/`.
Readable original reports, data, environment, prompt, training logs, and file
hashes are under `records/phi4-learning-20261112-181641-65125f/`.
No executed Colab notebook or full merged checkpoint was supplied.

Previous run artifacts and recovery workflows were removed during project
cleanup. Their removal inventory is `records/cleanup-20261112.json`.
