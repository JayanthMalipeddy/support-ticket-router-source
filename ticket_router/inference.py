from pathlib import Path
import time
from .domain import messages, decision, PROMPT_VERSION

class HFClassifier:
    def __init__(self,model_path):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM,AutoTokenizer
        if int(transformers.__version__.split('.')[0]) != 4:
            raise RuntimeError('Use the isolated notebook environment with Transformers 4; do not upgrade between training and evaluation')
        path=Path(model_path).resolve()
        if not (path/"config.json").is_file():
            raise ValueError("Existing local HF checkpoint required")
        self.tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(str(path),local_files_only=True,device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,attn_implementation="eager")
        self.model.eval(); self.identity=path.name
    def predict(self,text):
        import torch
        started=time.perf_counter()
        inputs=self.tokenizer.apply_chat_template(messages(text),tokenize=True,add_generation_prompt=True,
            return_tensors="pt",return_dict=True)
        ids=inputs['input_ids']
        if ids.shape[-1]>1000:
            raise ValueError("Ticket exceeds 1000-token limit")
        inputs={key:value.to(self.model.device) for key,value in inputs.items()}
        with torch.inference_mode():
            out=self.model.generate(**inputs,max_new_tokens=24,do_sample=False,use_cache=False,
                pad_token_id=self.tokenizer.pad_token_id,return_dict_in_generate=False)
        raw=self.tokenizer.decode(out[0,ids.shape[-1]:],skip_special_tokens=True).strip()
        return {**decision(raw),"raw_output":raw,"model_id":self.identity,"prompt_version":PROMPT_VERSION,
                "latency_ms":round((time.perf_counter()-started)*1000,2)}
