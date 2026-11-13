"""Build the self-contained, two-queue Phi-4-mini learning exercise."""

import json
from collections import Counter
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "phi4_lora_learning"
NOTEBOOK = ROOT / "notebooks" / "Phi4_Mini_LoRA_Learning.ipynb"

TRAIN = [
    ("Please give our new starter access to the shared mailbox called Finance. They already have access to the Finance network drive.", "O365"),
    ("My Outlook folders disappeared after my email profile was recreated. My documents and mapped drives are still available.", "O365"),
    ("The department shared drive opens normally, but Outlook says it cannot connect to the email server.", "O365"),
    ("A Teams meeting invitation arrived in my inbox, but accepting it does not add the meeting to my Outlook calendar.", "O365"),
    ("I can open the HR network folder, but I cannot send email from the HR shared mailbox. Please enable the mailbox sending permission.", "O365"),
    ("Outlook cannot expand the folders inside our shared mailbox. Windows Explorer opens our network folders without any problems.", "O365"),
    ("My headset works in other applications, but people cannot hear me during Teams calls.", "O365"),
    ("Outlook receives attachments correctly, but saving them to the department network folder gives an access-denied error. Saving to my desktop works.", "Fileservice"),
    (r"Someone posted a link in Teams to our network share at \\fileserver\Projects. Windows says I do not have permission to open that location.", "Fileservice"),
    ("The shared mailbox called Finance works. I need write permission on the separate Finance folder on the network file server.", "Fileservice"),
    (r"My manager emailed me the path \\fileserver\HR. The email opens correctly, but Windows cannot connect to that shared folder.", "Fileservice"),
    ("Our team stores exported email archives on the network drive. I accidentally deleted one from that drive and need the file restored from backup.", "Fileservice"),
    ("Outlook and Teams work after my laptop replacement, but the mapped department drive was not added. Please reconnect it.", "Fileservice"),
    ("The spreadsheet attached to a Teams message opens correctly. A separate copy on our department network share opens read-only, and I need permission to edit that copy.", "Fileservice"),
]

ORIGINAL_CHECK = [
    ("My work email stopped syncing on my phone.", "O365"),
    ("Teams won't let me join a scheduled meeting.", "O365"),
    ("I cannot open the shared mailbox for our support team.", "O365"),
    ("Windows says the network drive is unavailable.", "Fileservice"),
    ("Please give me permission to open the department's shared folder.", "Fileservice"),
    ("I accidentally deleted a document from our network share. Please recover it.", "Fileservice"),
]

# Author these before running the experiment; never pass them to the trainer.
FRESH_CHECK = [
    ("The room booking was accepted, but Outlook's calendar still shows the old meeting time. Please repair the calendar update.", "O365"),
    ("Messages to purchasing@company.example bounce because its shared mailbox is full. There is plenty of space on the purchasing network drive.", "O365"),
    ("Our project folder permissions are correct. A colleague needs permission to send email using the project's shared mailbox address.", "O365"),
    ("My Teams call is fine, but renaming a subfolder on the mapped R: drive says I lack the required permission.", "Fileservice"),
    ("The scans arrive in our shared mailbox successfully. Please recover last Friday's version of the invoice file stored on the network file server.", "Fileservice"),
    ("Please remove a departing contractor's access to the engineering network share. Their email access has already been handled.", "Fileservice"),
]

INSTRUCTION = (
    "Classify this support ticket.\n"
    "Return exactly one label: O365 or Fileservice.\n\n"
    "O365: Outlook, Teams, or mailbox problems.\n"
    "Fileservice: shared-drive or network-folder access, permissions, or file-recovery problems.\n\n"
    "Route according to what needs fixing."
)


def records(pairs):
    return [{"ticket": ticket, "label": label} for ticket, label in pairs]


def build():
    cells = []

    def md(source):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip() + "\n"})

    def code(source):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": dedent(source).strip() + "\n"})

    md("""
        # Learn LoRA with Phi-4-mini: two support queues

        This notebook trains **microsoft/Phi-4-mini-instruct** on your 14 labelled tickets.
        The base is loaded in 4-bit NF4 and only LoRA weights are trained: **QLoRA**.
        It does not modify your installed Ollama model.

        **Recorded successful run, 12 November 2026:** training-example accuracy
        improved from **12/14 to 14/14**; both sets of six checks remained **6/6**.
        The user repeated the trained-model check and again reported **14/14**.
        The project records preserve that run. This template includes later precision
        and reload checks, which have not yet been rerun on a GPU.

        **First:** select **Runtime > Change runtime type > T4 GPU** (or another NVIDIA GPU).
        Run numbered steps in order. After step 1, restart the session once, then start at step 2.

        The earlier **12/14** result used a batch prompt in Ollama. Here we establish a
        new baseline, with one ticket per request, and repeat that exact procedure after training.
        We retain all errors and invalid answers. No improved score is promised.

        The 14 training cases measure learning of known examples. The original six checks
        and six new checks stay out of training. The new checks are a tiny synthetic holdout,
        not evidence of production quality. Do not tune repeatedly against them.

        **Download the results in step 10 before Colab disconnects.** No Drive mount,
        source ZIP, Hugging Face upload, or account token is needed for public downloads.

        Sources: [Microsoft model](https://huggingface.co/microsoft/Phi-4-mini-instruct),
        [PEFT quantized LoRA](https://huggingface.co/docs/peft/developer_guides/quantization),
        [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer).
        Notebook code and data are checked locally. The successful training evidence
        came from the user's Colab session; local checks do not rerun GPU inference.
    """)
    md("""
        ## 1. Install the training libraries

        Run this cell once in a fresh GPU session. It keeps Colab's existing CUDA PyTorch.
        When installation finishes, choose **Runtime > Restart session**, then run **step 2**.
        Do not select 'Disconnect and delete runtime', which removes installed packages.
    """)
    code('''
        import subprocess
        import sys

        subprocess.run([
            sys.executable, "-m", "pip", "install", "--quiet",
            "transformers==4.57.6", "peft==0.18.1",
            "accelerate==1.11.0", "bitsandbytes==0.50.2",
        ], check=True)
        print("Installation finished. Restart the session once, then continue at step 2.")
    ''')
    md("""
        ## 2. Check the GPU and create a run folder

        Expected: CUDA is available and the GPU name is printed. If no GPU is available,
        stop and select a GPU runtime. A new run folder preserves earlier attempts.
    """)
    code('''
        import os
        os.environ["USE_TF"] = "0"
        os.environ["USE_FLAX"] = "0"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        import gc
        import json
        import math
        import platform
        import shutil
        import uuid
        from datetime import datetime, timezone
        from importlib.metadata import version
        from pathlib import Path

        import torch
        from packaging.version import Version
        from huggingface_hub import HfApi
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from peft.tuners.tuners_utils import BaseTunerLayer
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
            Trainer, TrainingArguments, default_data_collator, set_seed,
        )

        assert torch.cuda.is_available(), "Select Runtime > Change runtime type > GPU."
        assert Version(torch.__version__.split("+")[0]) >= Version("2.6"), "Use a fresh Colab GPU runtime with PyTorch >=2.6."
        expected_versions = {"transformers": "4.57.6", "peft": "0.18.1",
                             "accelerate": "1.11.0", "bitsandbytes": "0.50.2"}
        for package, expected in expected_versions.items():
            assert version(package) == expected, f"{package}: run step 1 and restart the session."
        set_seed(42)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        RUN = Path("/content") / f"phi4-learning-{stamp}-{uuid.uuid4().hex[:6]}"
        RUN.mkdir(parents=True, exist_ok=False)
        ADAPTER = RUN / "adapter"
        # Recent PyTorch includes emulated BF16 by default. Use native support
        # when choosing training precision, so a T4 selects FP16.
        USE_BF16 = torch.cuda.is_bf16_supported(including_emulation=False)
        COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

        def save_json(name, data):
            (RUN / name).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

        environment = {
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "compute_dtype": str(COMPUTE_DTYPE),
            **{name: version(name) for name in expected_versions},
        }
        save_json("environment.json", environment)
        print(json.dumps(environment, indent=2))
        print("Run folder:", RUN)
    ''')
    md("""
        ## 3. Prepare the examples

        **Only TRAIN enters the optimizer.** Tickets 1 and 5 have the correct O365 labels.
        Each record becomes a user ticket plus an assistant answer. Expected labels are
        never included in inference prompts. The same prompt function is used throughout.
    """)
    data_source = (
        "TRAIN = " + repr(records(TRAIN)) + "\n\n"
        "ORIGINAL_CHECK = " + repr(records(ORIGINAL_CHECK)) + "\n\n"
        "FRESH_CHECK = " + repr(records(FRESH_CHECK)) + "\n\n"
        "INSTRUCTION = " + repr(INSTRUCTION) + "\n\n"
    )
    code(data_source + dedent('''
        from collections import Counter

        def prompt_for(ticket):
            return f"{INSTRUCTION}\\n\\nTicket: {ticket}\\nLabel:"

        SETS = {"training_examples": TRAIN, "original_checks": ORIGINAL_CHECK,
                "fresh_holdout": FRESH_CHECK}
        assert len(TRAIN) == 14
        assert Counter(row["label"] for row in TRAIN) == {"O365": 7, "Fileservice": 7}
        assert len(ORIGINAL_CHECK) == len(FRESH_CHECK) == 6
        all_texts = [row["ticket"].strip().casefold() for rows in SETS.values() for row in rows]
        assert len(all_texts) == len(set(all_texts)), "Duplicate ticket across data sets."
        for name, rows in SETS.items():
            assert all(row["label"] in {"O365", "Fileservice"} for row in rows)
            (RUN / f"{name}.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\\n" for row in rows), encoding="utf-8")
        save_json("prompt.json", {"instruction": INSTRUCTION, "suffix": "\\n\\nTicket: <ticket>\\nLabel:"})
        print("Training:", len(TRAIN), "Original checks:", len(ORIGINAL_CHECK),
              "Fresh holdout:", len(FRESH_CHECK))
        print("Example training answer:", TRAIN[0]["label"])
        print(prompt_for(TRAIN[0]["ticket"]))
    '''))
    md("""
        ## 4. Download and load the original model

        The first download includes several GB of weights. Wait for it to finish.
        We record the exact Hugging Face revision and use the built-in Phi3 implementation
        in the pinned Transformers release. No custom model code is executed.
    """)
    code('''
        MODEL_ID = "microsoft/Phi-4-mini-instruct"
        REVISION = HfApi().model_info(MODEL_ID).sha
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=False)
        assert tokenizer.chat_template, "The model must supply its chat template."
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=COMPUTE_DTYPE,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, trust_remote_code=False,
            quantization_config=quantization, device_map={"": 0},
            dtype=COMPUTE_DTYPE, attn_implementation="sdpa",
        )
        assert getattr(model, "is_loaded_in_4bit", False)
        # Preparation also casts non-quantized weights. Do this before the baseline
        # so its base weights/dtypes match those used with the trained adapter.
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model.gradient_checkpointing_disable()
        model.config.use_cache = False
        STOP_IDS = [tokenizer.eos_token_id]
        end_id = tokenizer.get_vocab().get("<|end|>")
        if end_id is not None and end_id not in STOP_IDS:
            STOP_IDS.append(end_id)
        GENERATION = {
            "max_new_tokens": 12, "do_sample": False, "use_cache": False,
            "eos_token_id": STOP_IDS, "pad_token_id": tokenizer.pad_token_id,
        }
        save_json("provenance.json", {
            "model_id": MODEL_ID, "revision": REVISION, "generation": GENERATION,
            "quantization": "4-bit NF4, double quantization", "seed": 42,
            "evaluation": "one ticket per request; same prompt and decoding before/after",
        })
        print("Loaded:", MODEL_ID, "revision:", REVISION)
    ''')
    md("""
        ## 5. Measure the baseline before training

        Every request starts with just one ticket and no previous conversation. This is
        a new matched baseline; it may differ from the 12/14 Ollama batch result.
        Only an exact label, allowing surrounding whitespace, counts as correct.
        Inference exceptions stop the run instead of being counted as wrong answers.
    """)
    code('''
        @torch.inference_mode()
        def classify(ticket):
            model.eval()
            messages = [{"role": "user", "content": prompt_for(ticket)}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to("cuda:0")
            output = model.generate(**inputs, **GENERATION)
            new_ids = output[0, inputs["input_ids"].shape[1]:].tolist()
            # Remove only terminal stop tokens; retain any unexpected special tokens.
            while new_ids and new_ids[-1] in STOP_IDS:
                new_ids.pop()
            return tokenizer.decode(new_ids, skip_special_tokens=False).strip()

        def evaluate_all(stage):
            reports = {}
            for name, rows in SETS.items():
                cases = []
                for index, row in enumerate(rows, 1):
                    response = classify(row["ticket"])
                    cases.append({"id": index, **row, "response": response,
                                  "correct": response == row["label"]})
                    print(f"{name} {index}: {response!r} (expected {row['label']})")
                correct = sum(case["correct"] for case in cases)
                reports[name] = {"correct": correct, "total": len(cases),
                                 "accuracy": correct / len(cases), "cases": cases}
                # Save each completed set immediately, so it survives a later failure.
                save_json(f"{stage}.json", reports)
                print(f"{stage} / {name}: {correct}/{len(cases)} = {correct/len(cases):.1%}")
            return reports

        BEFORE = evaluate_all("before")
    ''')
    md("""
        ## 6. Tokenize the 14 training records

        The instruction and ticket tokens receive label **-100**, which excludes them
        from the loss. Only the assistant's correct label and end-of-answer tokens are
        supervised. Padding is excluded too. The code checks the chat-template boundary
        and refuses to silently truncate an example.
    """)
    code('''
        MAX_LENGTH = 256

        def encode_record(row):
            user = {"role": "user", "content": prompt_for(row["ticket"])}
            prefix = tokenizer.apply_chat_template([user], tokenize=True, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(
                [user, {"role": "assistant", "content": row["label"]}],
                tokenize=True, add_generation_prompt=False,
            )
            assert full[:len(prefix)] == prefix, "Training/inference template boundary differs."
            assert len(prefix) < len(full) <= MAX_LENGTH, "Check length; do not truncate the answer."
            targets = [-100] * len(prefix) + full[len(prefix):]
            padding = MAX_LENGTH - len(full)
            return {
                "input_ids": full + [tokenizer.pad_token_id] * padding,
                "attention_mask": [1] * len(full) + [0] * padding,
                "labels": targets + [-100] * padding,
            }

        class TicketDataset(torch.utils.data.Dataset):
            def __init__(self, rows):
                self.items = [encode_record(row) for row in rows]

            def __len__(self):
                return len(self.items)

            def __getitem__(self, index):
                return self.items[index]

        training_dataset = TicketDataset(TRAIN)
        supervised = [t for t in training_dataset[0]["labels"] if t != -100]
        print("Records used for training:", len(training_dataset))
        print("Supervised answer tokens:", tokenizer.decode(supervised, skip_special_tokens=False))
        assert tokenizer.decode(supervised, skip_special_tokens=True).strip() == "O365"
    ''')
    md("""
        ## 7. Attach the LoRA adapter

        Rank **8**, alpha **16**, dropout **0.05**. Phi-4-mini uses **qkv_proj** and
        **o_proj** for the attention projections. The printed trainable fraction should
        be small. Assertions verify that every trainable parameter belongs to LoRA.
    """)
    code('''
        config = LoraConfig(
            task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["qkv_proj", "o_proj"], bias="none",
        )
        model = get_peft_model(model, config)
        model.peft_config["default"].revision = REVISION
        model.print_trainable_parameters()
        initial_adapter = {name: parameter.detach().float().cpu().clone()
                           for name, parameter in model.named_parameters() if parameter.requires_grad}
        assert initial_adapter and all("lora_" in name for name in initial_adapter)
        assert all(not parameter.requires_grad for name, parameter in model.named_parameters()
                   if "lora_" not in name)
        print("Base weights are frozen. LoRA weights are ready to train.")
    ''')
    md("""
        ## 8. Train and save the adapter

        **This cell performs fine-tuning.** It runs three epochs over 14 examples,
        using batch size 1, gradient accumulation 2, and learning rate 0.0002.
        These are educational starting settings, not optimized hyperparameters.
        Loss measures prediction error on supervised training tokens; a lower loss does
        not prove better held-out accuracy. We verify that adapter weights actually changed.

        Run this training cell once per experiment. For another experiment, start a new
        run from step 2 so that the baseline and initialization are measured again.
    """)
    code('''
        assert not ADAPTER.exists(), "This run already has an adapter. Start a new run from step 2."
        args = TrainingArguments(
            output_dir=str(RUN / "trainer"), num_train_epochs=3,
            per_device_train_batch_size=1, gradient_accumulation_steps=2,
            learning_rate=2e-4, lr_scheduler_type="constant", warmup_ratio=0.0,
            weight_decay=0.0, max_grad_norm=1.0, optim="adamw_torch",
            fp16=not USE_BF16, bf16=USE_BF16,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=1, logging_nan_inf_filter=False,
            save_strategy="no", eval_strategy="no", report_to="none",
            seed=42, data_seed=42, dataloader_num_workers=0,
        )
        trainer = Trainer(
            model=model, args=args, train_dataset=training_dataset,
            data_collator=default_data_collator, processing_class=tokenizer,
        )
        result = trainer.train()
        assert trainer.state.global_step > 0
        assert math.isfinite(result.training_loss), "Nonfinite loss; do not use this adapter."
        delta = 0.0
        for name, parameter in model.named_parameters():
            if name in initial_adapter:
                current = parameter.detach().float().cpu()
                assert torch.isfinite(current).all(), f"Nonfinite adapter tensor: {name}"
                delta += (current - initial_adapter[name]).abs().sum().item()
        assert delta > 0, "No adapter weight changed."
        model.save_pretrained(str(ADAPTER), safe_serialization=True)
        tokenizer.save_pretrained(str(ADAPTER))
        assert (ADAPTER / "adapter_model.safetensors").stat().st_size > 0
        save_json("training.json", {
            "metrics": result.metrics, "logs": trainer.state.log_history,
            "optimizer_steps": trainer.state.global_step, "adapter_absolute_change": delta,
            "learning_rate": 2e-4, "epochs": 3, "lora_rank": 8,
            "lora_alpha": 16, "target_modules": ["qkv_proj", "o_proj"],
        })
        del initial_adapter
        print("Adapter saved:", ADAPTER)
        print("Optimizer steps:", trainer.state.global_step, "Adapter absolute change:", delta)
    ''')
    md("""
        ## 9. Reload the saved adapter and compare predictions

        We unload the in-memory adapter **without merging**, attach the saved one to
        the same frozen base, and evaluate again. The prompt, quantization, decoding,
        input order, and model revision stay the same. This also checks that the saved
        adapter can be read. No target score is enforced.
    """)
    code('''
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        base_model = model.unload()
        assert not any(isinstance(layer, BaseTunerLayer) for layer in base_model.modules()), "Unloading left adapter layers behind."
        assert not any("lora_" in name for name, _ in base_model.named_parameters()), "Unloading left adapter weights behind."
        # PEFT 0.18.1 unload removes the layers but leaves this metadata attribute.
        # Remove it only after verifying that no adapter layers or weights remain.
        if "peft_config" in vars(base_model):
            delattr(base_model, "peft_config")
        del model
        model = PeftModel.from_pretrained(base_model, str(ADAPTER), is_trainable=False)
        assert set(model.peft_config) == {"default"}, "Expected exactly one adapter."
        expected_layers = model.config.num_hidden_layers * len(model.peft_config["default"].target_modules)
        assert sum(isinstance(layer, BaseTunerLayer) for layer in model.modules()) == expected_layers
        model.gradient_checkpointing_disable()
        model.config.use_cache = False
        model.eval()
        AFTER = evaluate_all("after")
        comparison = {}
        for name in SETS:
            before, after = BEFORE[name], AFTER[name]
            comparison[name] = {
                "before_accuracy": before["accuracy"], "after_accuracy": after["accuracy"],
                "difference_percentage_points": 100 * (after["accuracy"] - before["accuracy"]),
            }
            print(f"{name}: BEFORE {before['correct']}/{before['total']} "
                  f"-> AFTER {after['correct']}/{after['total']}")
        save_json("comparison.json", comparison)
        print("Training-example improvements measure learning of known examples.")
        print("Check the holdout separately; six synthetic cases cannot establish production quality.")
    ''')
    md("""
        ## 10. Download your adapter, data, and evidence

        The ZIP contains adapter weights, tokenizer, model revision, exact prompt,
        before/after predictions, and loss logs. It does **not** contain the full base
        weights. Keep it before the Colab session expires. Loading this adapter later
        requires the recorded base model. Ollama remains a separate deployment step.

        If a later step fails after training, this download cell can still save the
        artifacts that were written successfully.
    """)
    code('''
        from google.colab import files

        archive = shutil.make_archive(str(RUN), "zip", root_dir=str(RUN))
        print("Download:", archive)
        files.download(archive)
    ''')

    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "colab": {"name": NOTEBOOK.name},
                     "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "cells": [{"id": f"learning-{i:02d}", **cell} for i, cell in enumerate(cells)],
    }
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    for name, pairs in [("train", TRAIN), ("original_checks", ORIGINAL_CHECK), ("fresh_holdout", FRESH_CHECK)]:
        (EXAMPLES / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records(pairs)), encoding="utf-8")
    assert Counter(label for _, label in TRAIN) == {"O365": 7, "Fileservice": 7}
    print(NOTEBOOK)
    print("14 training records; 6 original checks; 6 fresh held-out checks.")


if __name__ == "__main__":
    build()
