"""Real-run integrity checks and evidence export. No prediction fixtures."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError('Unsafe ZIP path')
        z.extractall(destination)


def verify_training(folder='artifacts/phi4-router-adapter'):
    from safetensors import safe_open
    import torch
    folder = Path(folder)
    state = read(folder / 'trainer_state.json')
    results = read(folder / 'train_results.json')
    if state['global_step'] <= 0 or state['global_step'] != state['max_steps']:
        raise ValueError('Training did not reach its recorded final step')
    if state['epoch'] < state['num_train_epochs']:
        raise ValueError('Training did not complete its configured epochs')
    if not math.isfinite(results['train_loss']):
        raise ValueError('Training loss is not finite')
    weights = folder / 'adapter_model.safetensors'
    count, parameters, nonzero_b = 0, 0, 0
    with safe_open(weights, framework='pt', device='cpu') as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f'Nonfinite adapter tensor: {key}')
            count += 1
            parameters += tensor.numel()
            if 'lora_B' in key:
                nonzero_b += torch.count_nonzero(tensor).item()
    if not count or not nonzero_b:
        raise ValueError('Adapter is missing learned LoRA B weights')
    report = {'global_step': state['global_step'], 'epochs': state['epoch'],
              'train_runtime_seconds': results['train_runtime'], 'train_loss': results['train_loss'],
              'tensor_count': count, 'adapter_parameters': parameters, 'nonzero_lora_b': nonzero_b,
              'adapter_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
              'note': 'Training completion and weight integrity only; not validation accuracy.'}
    write('artifacts/training-verification.json', report)
    print(json.dumps(report, indent=2))


def verify_merge():
    from transformers import AutoTokenizer
    from ticket_router.domain import messages
    import torch
    from safetensors import safe_open
    folder = Path('artifacts/phi4-router-merged')
    base = Path('artifacts/base-snapshot-path.txt').read_text().strip()
    baseline = AutoTokenizer.from_pretrained(base, local_files_only=True)
    merged = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    if not merged.chat_template:
        raise ValueError('Exported tokenizer has no chat template')
    rows = read('data/prepared/validation.json')
    for row in rows:
        args = dict(tokenize=True, add_generation_prompt=True, return_dict=False)
        if baseline.apply_chat_template(messages(row['text']), **args) != merged.apply_chat_template(messages(row['text']), **args):
            raise ValueError('Base/merged inference prompt token IDs differ')
    index = folder / 'model.safetensors.index.json'
    if index.exists():
        weight_map = read(index)['weight_map']
        shards = sorted(set(weight_map.values()))
    else:
        weight_map, shards = None, ['model.safetensors']
    tensor_count = 0
    for name in shards:
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file():
            raise ValueError(f'Missing or unsafe exported shard: {name}')
        with safe_open(path, framework='pt', device='cpu') as f:
            keys = set(f.keys())
            if weight_map and keys != {k for k, v in weight_map.items() if v == name}:
                raise ValueError('Exported shard does not match weight index')
            for key in keys:
                if not torch.isfinite(f.get_tensor(key)).all().item():
                    raise ValueError(f'Nonfinite merged tensor: {key}')
            tensor_count += len(keys)
    report = {'shards': shards, 'tensor_count': tensor_count, 'prompt_parity_cases': len(rows),
              'note': 'Export structure, finite weights and prompt parity; model quality is measured separately.'}
    write('artifacts/merge-verification.json', report)
    print(json.dumps(report, indent=2))


def smoke(model):
    from ticket_router.inference import HFClassifier
    from ticket_router.domain import LABELS
    rows = read('data/prepared/validation.json')
    # One real held-out ticket per class; expected labels never enter predict().
    selected = [next(r for r in rows if r['label'] == label) for label in LABELS]
    classifier = HFClassifier(model)
    results = [{'id': row['id'], 'expected': row['label'], **classifier.predict(row['text'])}
               for row in selected]
    write('artifacts/smoke.json', results)
    print(json.dumps(results, indent=2))
    if not any(r['label'] in LABELS for r in results):
        raise RuntimeError('No valid queue labels in smoke outputs; inspect prompt/model before evaluation')


def evidence():
    # May also be called after failure; never manufactures missing result files.
    with zipfile.ZipFile('week5-evidence.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for folder in ['artifacts', 'data/prepared', 'ticket_router', 'scripts', 'configs', 'notebooks']:
            for path in Path(folder).rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts and path.suffix in {
                        '.json', '.jsonl', '.yaml', '.txt', '.log', '.png', '.jinja', '.py', '.ipynb'}:
                    z.write(path, path.as_posix())
        z.write('pyproject.toml')
    print('week5-evidence.zip (includes ticket data, code and diagnostic logs; keep private)')


def environment(stage):
    import importlib.metadata as metadata
    import torch
    versions = {p: metadata.version(p) for p in ['torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes', 'trl']}
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in the isolated environment')
    if versions['transformers'].split('.')[0] != '4':
        raise RuntimeError('Unexpected Transformers major version')
    initial = Path('artifacts/environment-initial.json')
    if initial.exists() and read(initial)['packages'] != versions:
        raise RuntimeError('Package versions changed during the run')
    report = {'packages': versions, 'python': sys.version, 'executable': sys.executable,
              'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda}
    write(f'artifacts/environment-{stage}.json', report)
    Path(f'artifacts/environment-{stage}.txt').write_text(
        subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['verify-training', 'verify-merge', 'smoke', 'evidence', 'environment'])
    parser.add_argument('--model', default='artifacts/phi4-router-merged')
    parser.add_argument('--stage', default='initial')
    args = parser.parse_args()
    if args.command == 'verify-training': verify_training()
    elif args.command == 'verify-merge': verify_merge()
    elif args.command == 'smoke': smoke(args.model)
    elif args.command == 'evidence': evidence()
    else: environment(args.stage)
