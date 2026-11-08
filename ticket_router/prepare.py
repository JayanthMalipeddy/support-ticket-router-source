import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from .domain import LABELS, SYSTEM, fingerprint

def prepare(source, output, text_column, label_column, seed=42):
    source, output = Path(source), Path(output)
    unique, duplicates = {}, 0
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {text_column,label_column} <= set(reader.fieldnames or []):
            raise ValueError(f"Required columns missing; available: {reader.fieldnames}")
        for line,row in enumerate(reader,2):
            text,label = (row.get(text_column) or "").strip(), (row.get(label_column) or "").strip()
            if not text or label not in LABELS:
                raise ValueError(f"Empty text or unknown label at CSV line {line}")
            key = fingerprint(" ".join(text.casefold().split()))
            if key in unique:
                if unique[key]["label"] != label:
                    raise ValueError(f"Conflicting duplicate at CSV line {line}")
                duplicates += 1
            else:
                unique[key] = {"id":key,"text":text,"label":label}
    train,validation,counts = [],[],{}
    for label in LABELS:
        rows = sorted((r for r in unique.values() if r["label"] == label),key=lambda r:r["id"])
        if len(rows)<5:
            raise ValueError(f"Need at least 5 unique examples for {label}; got {len(rows)}")
        random.Random(f"{seed}:{label}").shuffle(rows)
        n = max(1,round(len(rows)*.2))
        validation.extend(rows[:n]); train.extend(rows[n:])
        counts[label] = {"train":len(rows)-n,"validation":n}
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be empty; preserve the existing split")
    output.mkdir(parents=True,exist_ok=True)
    def write(name,value):
        (output/name).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding="utf-8")
    write("train.json",train); write("validation.json",validation)
    write("support_tickets.json",[{"system":SYSTEM,"conversations":[{"from":"human","value":r["text"]},{"from":"gpt","value":r["label"]}]} for r in train])
    write("dataset_info.json",{"support_tickets":{"file_name":"support_tickets.json","formatting":"sharegpt","columns":{"messages":"conversations","system":"system"}}})
    manifest = {"schema_version":"1","seed":seed,"counts":counts,"duplicates_removed":duplicates,
                "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"train_fingerprint":fingerprint(train),"validation_fingerprint":fingerprint(validation)}
    write("manifest.json",manifest)
    return manifest

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("source"); p.add_argument("--output",default="data/prepared")
    p.add_argument("--text-column",required=True); p.add_argument("--label-column",required=True)
    a=p.parse_args(); print(json.dumps(prepare(a.source,a.output,a.text_column,a.label_column),indent=2))
