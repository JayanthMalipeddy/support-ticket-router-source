# Phi-4-mini support-ticket routing: successful LoRA run

The current experiment routes tickets to **O365** or **Fileservice**.
On 12 November 2026, QLoRA training completed in Google Colab on a Tesla T4.
The saved adapter corrected both training-example errors. The user repeated
the trained-model check and again obtained **14/14**.

| Dataset | Before training | After training |
|---|---:|---:|
| Training examples | 12/14 (85.7%) | 14/14 (100%) |
| Original checks, excluded from training | 6/6 | 6/6 |
| Fresh synthetic holdout, excluded from training | 6/6 | 6/6 |

This demonstrates learning on known training examples. No held-out accuracy
improvement was measured; both tiny check sets already scored 100%.

## Current records

- [Latest-run pointer](records/latest.json): model, revision, scores, and artifact paths.
- [Successful-run evidence](records/phi4-learning-20261112-181641-65125f/): original reports, data, prompt, environment, training logs, and verification manifest.
- [Results review](docs/RESULTS_REVIEW.md): interpretation and practical limits.
- [Verification](docs/VERIFICATION.md): checks actually performed.
- [Cleanup record](records/cleanup-20261112.json): superseded resources removed from this workspace.

The original successful ZIP is preserved as
`artifacts/phi4-learning-20261112-181641-65125f/run.zip`.
Its extracted adapter and tokenizer are in the adjacent `adapter/` folder.
These binary artifacts are local and excluded from Git and the source ZIP.
Keep the run ZIP when backing up or transferring the project.

## Run the learning exercise

Upload [Phi4_Mini_LoRA_Learning.ipynb](notebooks/Phi4_Mini_LoRA_Learning.ipynb)
to Google Colab, select a GPU runtime, and follow its ten numbered steps.
The 14 training examples and both check sets are embedded; no source ZIP is
required. See [notebook instructions](docs/NOTEBOOK_GUIDE.md).

Model: `microsoft/Phi-4-mini-instruct`, base revision
`cfbefacb99257ffa30c83adab238a50856ac3083`.
Training used a 4-bit NF4 base with rank-8 LoRA, alpha 16, attention projections
`qkv_proj` and `o_proj`, three epochs, and 21 optimizer steps. Only adapter
weights were trained: 3,670,016 parameters, about 0.0956% of the adapted model.

The current notebook includes post-run precision and adapter-reload checks.
The successful run's recorded configuration is retained exactly as supplied.
An executed notebook was not included in the downloaded archive.

## Verify or package the project

```powershell
python -B scripts/record_learning_run.py --verify
python -B -m unittest discover -s tests -v
python -B -m scripts.package_source
```

The verification command checks the original ZIP hash, extracted-file hashes,
data/report consistency, and adapter tensor values without loading the model.
The source archive is `dist/support-ticket-router-source.zip`; it includes
notebooks, data, source, and readable records, and excludes model binaries.
Rebuild the notebook and example files with
`python -B scripts/build_phi4_learning_notebook.py`.

## API and integration source

`ticket_router/`, the integration examples, and their tests retain the reusable
local API/evaluation code. That API currently defines a broader seven-label
contract and has not been wired to this two-label adapter. The successful
measurements above were produced by the Colab notebook. No merged model or
Ollama import is included in this run. See [integration reference](docs/INTEGRATION.md).

Sources: [Microsoft model](https://huggingface.co/microsoft/Phi-4-mini-instruct),
[PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization).
