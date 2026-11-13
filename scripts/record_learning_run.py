"""Import and verify a saved two-queue Phi learning run without loading a model."""

import argparse
import hashlib
import json
import math
import shutil
import struct
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "phi4-learning-20261112-181641-65125f"
EXPECTED_SHA256 = "e0bb758433be4aa9318899696bd15d26fbacde40d8ccd50e2654671165d8c3b0"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def inspect_archive(path):
    payload = path.read_bytes()
    if digest(payload) != EXPECTED_SHA256:
        raise ValueError("Archive does not match the reviewed successful run")
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP checksum failure")
        members = {}
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            relative = PurePosixPath(entry.filename)
            if relative.is_absolute() or ".." in relative.parts or "\\" in entry.filename:
                raise ValueError("Unsafe archive member")
            if entry.filename in members or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Duplicate member or symbolic link")
            members[entry.filename] = archive.read(entry)

    def read(name):
        return json.loads(members[name])

    before, after, comparison = read("before.json"), read("after.json"), read("comparison.json")
    rows_by_group = {}
    scores = {}
    file_names = {"training_examples": "train", "original_checks": "original_checks", "fresh_holdout": "fresh_holdout"}
    for group, expected_size in [("training_examples", 14), ("original_checks", 6), ("fresh_holdout", 6)]:
        rows = [json.loads(line) for line in members[f"{group}.jsonl"].decode().splitlines()]
        canonical = [json.loads(line) for line in (ROOT / "examples" / "phi4_lora_learning" / f"{file_names[group]}.jsonl").read_text(encoding="utf-8").splitlines()]
        if rows != canonical or len(rows) != expected_size:
            raise ValueError(f"Unexpected data: {group}")
        if Counter(row["label"] for row in rows) != {"O365": expected_size // 2, "Fileservice": expected_size // 2}:
            raise ValueError(f"Unbalanced labels: {group}")
        for report in (before[group], after[group]):
            if [(c["ticket"], c["label"]) for c in report["cases"]] != [(r["ticket"], r["label"]) for r in rows]:
                raise ValueError(f"Prediction/data mismatch: {group}")
            correct = sum(c["response"] == c["label"] for c in report["cases"])
            if report["correct"] != correct or report["total"] != expected_size or report["accuracy"] != correct / expected_size:
                raise ValueError(f"Inconsistent metrics: {group}")
            if any(c["correct"] != (c["response"] == c["label"]) for c in report["cases"]):
                raise ValueError(f"Inconsistent case flags: {group}")
        expected_comparison = {
            "before_accuracy": before[group]["accuracy"], "after_accuracy": after[group]["accuracy"],
            "difference_percentage_points": 100 * (after[group]["accuracy"] - before[group]["accuracy"]),
        }
        if comparison[group] != expected_comparison:
            raise ValueError(f"Inconsistent comparison: {group}")
        scores[group] = {"total": expected_size, "before_correct": before[group]["correct"],
                         "after_correct": after[group]["correct"],
                         "corrected_ids": [c["id"] for c, a in zip(before[group]["cases"], after[group]["cases"]) if not c["correct"] and a["correct"]]}
        rows_by_group[group] = rows
    all_tickets = [r["ticket"].strip().casefold() for rows in rows_by_group.values() for r in rows]
    if len(all_tickets) != len(set(all_tickets)):
        raise ValueError("Duplicate ticket across splits")
    weights = members["adapter/adapter_model.safetensors"]
    header_size = struct.unpack("<Q", weights[:8])[0]
    tensors = {k: v for k, v in json.loads(weights[8:8 + header_size]).items() if k != "__metadata__"}
    elements = sum(math.prod(t["shape"]) for t in tensors.values())
    nonzero_b = 0
    for name, tensor in tensors.items():
        start, end = tensor["data_offsets"]
        if tensor["dtype"] != "F32" or end - start != math.prod(tensor["shape"]) * 4:
            raise ValueError(f"Unexpected tensor layout: {name}")
        data = weights[8 + header_size + start:8 + header_size + end]
        nonzero = False
        for (value,) in struct.iter_unpack("<f", data):
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite tensor: {name}")
            nonzero |= value != 0
        if "lora_B" in name and nonzero:
            nonzero_b += 1
    training, config, provenance = read("training.json"), read("adapter/adapter_config.json"), read("provenance.json")
    if len(tensors) != 128 or elements != 3670016 or nonzero_b != 64:
        raise ValueError("Unexpected adapter dimensions or unchanged B weights")
    if training["optimizer_steps"] != 21 or training["adapter_absolute_change"] <= 0:
        raise ValueError("Missing training updates")
    if not math.isfinite(training["metrics"]["train_loss"]) or any(not math.isfinite(row["loss"]) for row in training["logs"] if "loss" in row):
        raise ValueError("Nonfinite training loss")
    if config["r"] != 8 or config["lora_alpha"] != 16 or set(config["target_modules"]) != {"qkv_proj", "o_proj"}:
        raise ValueError("Unexpected adapter configuration")
    if config["base_model_name_or_path"] != provenance["model_id"] or config["revision"] != provenance["revision"]:
        raise ValueError("Inconsistent base-model identity")
    return members, {
        "run_id": RUN_ID, "archive_sha256": EXPECTED_SHA256, "archive_crc_passed": True,
        "model_id": provenance["model_id"], "base_revision": provenance["revision"],
        "scores": scores, "optimizer_steps": 21, "epochs": training["epochs"],
        "train_loss": training["metrics"]["train_loss"],
        "adapter_parameters": elements, "adapter_tensors": len(tensors),
        "nonzero_lora_B_matrices": nonzero_b, "all_adapter_values_finite": True,
        "adapter_absolute_change": training["adapter_absolute_change"],
        "local_model_inference_rerun": False,
    }


def import_run(archive):
    members, review = inspect_archive(archive)
    record_dir = ROOT / "records" / RUN_ID
    artifact_dir = ROOT / "artifacts" / RUN_ID
    if record_dir.exists() or artifact_dir.exists():
        raise FileExistsError("Run already imported; use --verify")
    record_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    shutil.copyfile(archive, artifact_dir / "run.zip")
    inventory = []
    for name, data in members.items():
        destination = artifact_dir / name if name.startswith("adapter/") else record_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        inventory.append({"archive_member": name, "path": destination.relative_to(ROOT).as_posix(),
                          "bytes": len(data), "sha256": digest(data)})
    # A small config copy makes the source-controlled record readable without weights.
    config_path = record_dir / "adapter_config.json"
    config_path.write_bytes(members["adapter/adapter_config.json"])
    inventory.append({"archive_member": "adapter/adapter_config.json", "path": config_path.relative_to(ROOT).as_posix(),
                      "bytes": config_path.stat().st_size, "sha256": digest(config_path.read_bytes())})
    write_json(record_dir / "verification.json", {**review, "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
               "method": "Offline archive integrity, data/report reconciliation, and finite/nonzero adapter tensor inspection"})
    write_json(record_dir / "manifest.json", {
        "run_id": RUN_ID, "archive_path": (artifact_dir / "run.zip").relative_to(ROOT).as_posix(),
        "archive_sha256": EXPECTED_SHA256, "files": inventory,
        "executed_notebook_supplied": False,
        "notebook_note": "The local notebook template includes post-run fixes; it is not an executed notebook from this run.",
    })
    write_json(ROOT / "records" / "latest.json", {**review,
               "records_path": record_dir.relative_to(ROOT).as_posix(),
               "adapter_path": (artifact_dir / "adapter").relative_to(ROOT).as_posix(),
               "archive_path": (artifact_dir / "run.zip").relative_to(ROOT).as_posix(),
               "scope": ["O365", "Fileservice"], "ollama_imported": False, "merged_model_supplied": False})
    print(f"Imported successful run: {record_dir}")


def verify_run():
    latest = json.loads((ROOT / "records" / "latest.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / latest["records_path"] / "manifest.json").read_text(encoding="utf-8"))
    _, review = inspect_archive(ROOT / manifest["archive_path"])
    for key, value in review.items():
        if latest[key] != value:
            raise ValueError(f"Latest-run summary differs from evidence: {key}")
    for item in manifest["files"]:
        path = (ROOT / item["path"]).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError("Record path escapes project")
        data = path.read_bytes()
        if len(data) != item["bytes"] or digest(data) != item["sha256"]:
            raise ValueError(f"Saved file differs from original archive: {item['path']}")
    print(f"PASS: {RUN_ID}; archive, {len(manifest['files'])} saved files, all scores, and adapter tensors verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--archive", type=Path, help="Import the reviewed successful ZIP")
    choice.add_argument("--verify", action="store_true", help="Verify the saved latest run")
    args = parser.parse_args()
    if args.archive:
        import_run(args.archive)
    else:
        verify_run()
