# Implementation verification

**Update, 6 November 2026:** user-supplied Colab training/evaluation artifacts have now been reviewed. Training completion is supported, but evaluation failed on all cases. See [RESULTS_REVIEW.md](RESULTS_REVIEW.md). The local checks and execution limitations below describe the earlier 5 November session.

## Corrected notebook checks — 6 November 2026

The revised notebook and source add isolated subprocess execution, pinned core dependencies, recovery from the saved adapter, real training/export integrity checks, dataset-derived smoke cases, strict evaluation failures and complete diagnostic export. See [NOTEBOOK_GUIDE.md](NOTEBOOK_GUIDE.md).

- Nine software tests passed, including preservation of chained inference errors and rejection of failed evaluation reports. These tests do not measure model quality.
- Fifteen Python source files, twelve notebook code cells and their embedded subprocess scripts passed syntax parsing. The generated notebook contains no execution outputs or fabricated metrics.
- The new training integrity checker was actually run locally against the supplied adapter weights: 177 completed steps, three epochs, 256 finite tensors, 11,534,336 parameters and 7,077,888 nonzero LoRA B values. The adapter SHA-256 is `50d04779b51d5e290bf1ecb83c0ce8a399f9ed6a6347af0faec5b476f1d83a6e`.
- The actual saved adapter tokenizer was loaded locally. Explicit dictionary tokenization returned `input_ids` and `attention_mask` and retained the system classification prompt. This tokenizer-only check used the installed local Transformers version; it does not validate pinned Colab installation or model generation.
- Fresh isolated installation, new merge verification, real smoke inference and corrected baseline/candidate evaluation still require execution on a GPU runtime. The old AttributeError's exact cause remains unconfirmed without its traceback.

Local checks on 5 November 2026:

- Seven unittest checks pass: API authentication/unavailable-model behavior, strict response and human-review handling, input validation, deterministic split and duplicate conflict rejection, invalid-prediction accounting, mismatched comparison rejection, and integration adapter separation.
- Model calls in those tests use a named fixture. No live model quality is measured.
- OpenAPI is generated from the actual application schemas.
- Colab notebook cells are syntax-checked locally; GPU installation, training, merge and real validation inference have not run.

Postman CLI fails at startup because a managed Private Cloud manifest cannot be read. No Postman workspace was initialized or published. Tests use the real FastAPI application through its in-process HTTP test client.

Extraction was verified at `C:\Projects\support-ticket-router-source` on 5 November 2026. `C:\Projects\support-ticket-router` was not present. The current source no longer relies on the earlier temporary staging directory. The packaging command now writes `dist/support-ticket-router-source.zip` within this workspace. No Week 3 or Week 4 files were changed.

## Training execution review

- Local Python 3.13 has PyTorch `2.14.1+cpu`, no CUDA build, `cuda.is_available() == False`, and zero CUDA devices. `nvidia-smi` is not on PATH; Windows GPU inventory through CIM was denied. This establishes no usable GPU in this Python environment, not the physical absence of GPU hardware.
- Installed global Transformers is `5.19.0`, outside this project's `<5` inference requirement. It was not changed because other projects may use that installation. Use an isolated environment for inference/training.
- No connected Colab or remote GPU execution tool is available in this session. A direct Python dataset download was refused by the network environment. No dataset or checkpoint download, fresh dependency install, GPU preflight, training, merge, live smoke test or model benchmark was completed here.
- The notebook now resolves LLaMA Factory and `.[inference]` together, checks dependency consistency and CUDA, and supports selecting a recorded upstream commit. It still defaults to `main`; its resolved commit and package list must accompany any result. Fresh Colab compatibility remains unverified.
- Upstream review confirmed `phi4_mini` in [current template source](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/template.py). The [current dependency declaration](https://github.com/hiyouga/LLaMA-Factory/blob/main/pyproject.toml) permits Transformers 5, whereas this project's inference path requires Transformers below 5; joint resolution enforces that boundary. Older v0.9.3/v0.9.4 template sources reviewed did not contain `phi4_mini`, so they were not substituted as a purported compatible pin.
- Preflight now uses the trainer's template encoder and resolved cutoff, validates split/ShareGPT integrity and prompt parity, and exercises bitsandbytes on CUDA. These new GPU checks are syntax-checked only until a GPU runtime executes them.
- Authored smoke-example classification mistakes no longer prevent held-out evaluation. Actual inference exceptions still stop the cell. The notebook offers adapter download in addition to merged weights.

## Resume in a GPU runtime

1. Run `python -m scripts.package_source` locally.
2. Open `notebooks/Train_Phi4_Support_Router.ipynb` in Colab, select a GPU runtime, and upload `dist/support-ticket-router-source.zip` in the first cell.
3. Run installation and preflight cells; stop on conflicts or errors. Retain their output when diagnosing failures. Do not disable checks to force training through.
4. Run training, merge, smoke, baseline and candidate evaluation. Preserve comparison reports even if the candidate is worse.
5. Download the executed notebook, evidence ZIP, adapter and merged weights before the runtime expires. Record measured results separately from fixture tests. Publish and submit manually only when ready.

The older applications are integration-ready through the supplied contract, client and exchange adapters, but their screens/database importers are not wired to this component. Training must produce a real checkpoint before the model-backed route is available.
