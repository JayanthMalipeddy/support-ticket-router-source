"""Run in Colab after installing LLaMA Factory and preparing data."""
import copy
import json
from pathlib import Path
import torch
import yaml
from transformers import AutoTokenizer
from llamafactory.data.template import TEMPLATES
from ticket_router.domain import SYSTEM,fingerprint,messages

assert torch.cuda.is_available(), 'Select a GPU runtime first'
assert 'phi4_mini' in TEMPLATES, 'This LLaMA Factory checkout lacks Phi-4-mini support'
config=yaml.safe_load(Path('artifacts/train-resolved.yaml').read_text())
snapshot=config['model_name_or_path']
tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
template=copy.deepcopy(TEMPLATES[config['template']])
template.fix_special_tokens(tokenizer)
folder=Path(config['dataset_dir'])
rows=json.loads((folder/'train.json').read_text(encoding='utf-8'))
validation=json.loads((folder/'validation.json').read_text(encoding='utf-8'))
manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
for name,split in [('train',rows),('validation',validation)]:
    assert fingerprint(split)==manifest[name+'_fingerprint'], f'{name} fingerprint mismatch'
expected=[{'system':SYSTEM,'conversations':[{'from':'human','value':r['text']},{'from':'gpt','value':r['label']}]} for r in rows]
assert json.loads((folder/'support_tickets.json').read_text(encoding='utf-8'))==expected, 'ShareGPT records differ from fingerprinted training split'
too_long=[]
lengths=[]
for row in rows:
    prompt,target=template.encode_oneturn(tokenizer,[{'role':'user','content':row['text']},{'role':'assistant','content':row['label']}],system=SYSTEM)
    length=len(prompt)+len(target)+int(template.efficient_eos)
    lengths.append(length)
    if length>config['cutoff_len']:
        too_long.append(row['id'])
    assert prompt==tokenizer.apply_chat_template(messages(row['text']),tokenize=True,add_generation_prompt=True), 'Training/inference prompt token IDs differ'
assert not too_long, f'{len(too_long)} training tickets exceed cutoff; curate or raise cutoff before training'
assert all(len(tokenizer.apply_chat_template(messages(r['text']),tokenize=True,add_generation_prompt=True))<=1000 for r in validation), 'Validation tickets exceed inference limit; resolve before training'
# Exercise the actual CUDA quantization backend before starting training.
import bitsandbytes as bnb
layer=bnb.nn.Linear4bit(64,64,bias=False,compute_dtype=torch.float16,quant_type='nf4').to('cuda')
with torch.no_grad():
    layer(torch.zeros(1,64,device='cuda',dtype=torch.float16))
torch.cuda.synchronize()
report={'gpu':torch.cuda.get_device_name(0),'vram_bytes':torch.cuda.get_device_properties(0).total_memory,
        'torch':torch.__version__,'cuda':torch.version.cuda,'training_examples':len(rows),
        'validation_examples':len(validation),'max_training_tokens':max(lengths),
        'template':config['template'],'prompt_parity':True,'cuda_4bit_operation':True}
Path('artifacts/preflight.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
