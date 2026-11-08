# Support Ticket Router — Week 5

Independent Phi-4-mini project: CSV preparation, QLoRA training through LLaMA Factory, merge, paired classification evaluation, and an advisory local API for Weeks 3 and 4.

**Status (6 November 2026):** supplied Colab artifacts record completed training and include adapter weights. Both held-out evaluation runs failed on all 118 cases with `AttributeError`; routing quality and speed improvement remain unverified. See [results review](docs/RESULTS_REVIEW.md) for evidence and recovery steps. Phi-4-mini is a custom substitution for the handout's Qwen model. The Colab notebook is the training entry point.

## Start with Colab

Verified extracted working location: `C:\Projects\support-ticket-router-source`. The previously requested `C:\Projects\support-ticket-router` folder was not found. Commands below run from the extracted folder; it no longer depends on the temporary staging directory.

Build the upload archive with `python -m scripts.package_source`. It writes `dist/support-ticket-router-source.zip` inside this workspace, excluding datasets, model artifacts and virtual environments.

Open `notebooks/Train_Phi4_Support_Router.ipynb` in Google Colab. Select a GPU runtime and upload the newly generated source ZIP when prompted. Choose **recover** to reuse your existing adapter and original evidence ZIP, or **train** for a new training run. See [notebook instructions](docs/NOTEBOOK_GUIDE.md). Every run uses a separate working directory and an isolated Python environment for all ML subprocesses.

The initial configuration is 4-bit QLoRA, rank 8, batch 1, gradient accumulation 8, FP16, 1,024-token cutoff and three epochs. These are starting settings, not a verified T4 benchmark. Actual memory usage depends on the runtime. Merge runs on CPU and needs sufficient RAM/disk for unquantized weights. Do not enable quantization in the merge configuration.

The model is `microsoft/Phi-4-mini-instruct`, using LLaMA Factory's **phi4_mini** template. Your Ollama GGUF file is not used as the training checkpoint. Package versions and upstream commit are recorded by the notebook; the fresh Colab installation remains to be executed.

The installer pins core dependencies in `configs/colab-constraints.txt`, resolves trainer and inference requirements together, then runs `pip check` and checks CUDA. It uses the recorded factory commit instead of a moving branch. Preflight verifies split fingerprints, ShareGPT contents, training-template token lengths, training/inference prompt parity and a real CUDA 4-bit operation. Training completion and merged-weight checks inspect actual files. Smoke inference uses real validation tickets, with expected labels kept outside model input. An entirely invalid smoke run stops for investigation; individual classification mistakes remain in the measured results. Save the adapter and merged model before the runtime expires. The revised isolated installation still requires a fresh GPU verification run.

The current local Python reports CPU-only PyTorch and zero CUDA devices. This session cannot execute the GPU workflow; open the notebook in an accessible Colab GPU runtime. See `docs/VERIFICATION.md` for the exact limits of the checks performed.

## Local setup

From this folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
# Only when you have downloaded the merged HF model:
.\.venv\Scripts\python.exe -m pip install -e '.[inference]'
$env:ROUTER_MODEL_PATH = 'C:\Models\phi4-router-merged'
$env:ROUTER_API_TOKEN = '<your-private-random-token>'
& .\scripts\start.ps1
```

Port **8025** avoids the existing projects' 8010 port. The API binds to loopback. It does not store ticket text or call cloud inference. `GET /health` reports readiness; `POST /v1/route` requires a bearer token. Without a model it returns 503, never fabricated predictions. Overlong inputs return 422; busy inference returns 429. No calibrated confidence is claimed. Unknown output returns a null label and requires human review; the genuine Support general class also requires review.

## Dataset and evaluation

```powershell
python -m ticket_router.prepare data/raw/support_tickets.csv --text-column '<actual text column>' --label-column '<actual label column>'
python -m ticket_router.evaluate run --model artifacts/base-snapshot --output artifacts/baseline.json
python -m ticket_router.evaluate run --model artifacts/phi4-router-merged --output artifacts/tuned.json
python -m ticket_router.evaluate compare artifacts/baseline.json artifacts/tuned.json --output artifacts/comparison.json
```

Preparation validates all seven classes, removes case/whitespace-normalized duplicates, rejects conflicting labels, and uses a deterministic approximately 80/20 per-class split. Split fingerprints and class counts are saved. It refuses to overwrite a prepared dataset. Near-duplicate paraphrases are not automatically detected; inspect dataset quality before claiming generalization.

Only training rows enter ShareGPT JSON. The held-out split is not passed to training. Both models receive the same exact-label prompt and deterministic, cache-disabled decoding. Invalid generated outputs count as incorrect. The evaluation CLI now stops on inference exceptions with a full traceback instead of writing a completed zero-score report; comparison rejects reports containing exceptions. This differs from the academy's letter-choice baseline and is explicitly an identical-prompt Phi experiment. If you tune repeatedly against the validation split, reserve a new untouched test set before final claims.

Reports include accuracy, macro-F1, per-class precision/recall/F1, confusion matrix, per-case predictions and median observed latency. Timing excludes model loading but includes tokenization and generation; it includes the first generation, so use the same environment and order policy for both models. Cost savings are not inferred from parameter count.

## Join with Week 3 and Week 4

See `docs/INTEGRATION.md`. The reusable Python client and adapters are in `ticket_router/integrations.py`. The API contract is `docs/openapi.json`. Both older applications remain unchanged; their final UI/storage wiring is a separate consumer change. There is no dependency on either installation.

## Verify and submit

```powershell
python -m unittest discover -s tests -v
python -m scripts.export_contract
```

Tests use explicit fixtures and do not establish model quality. For submission retain the successful notebook output, training loss plot, adapter/merged model provenance, smoke results, baseline/tuned reports and confusion matrix. Record a Loom demo and submit the custom repository link manually. No publication or submission is automated.

Sources: [Microsoft model card](https://huggingface.co/microsoft/Phi-4-mini-instruct), [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory), [academy dataset and notebook](https://github.com/The-Gen-Academy/5A-Fine-Tune-a-Support-Ticket-Router). The upstream dataset is downloaded at execution time; inspect its usage terms before redistributing it.
