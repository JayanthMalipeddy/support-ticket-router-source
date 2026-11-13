from pathlib import Path
import time
from .domain import messages, decision, PROMPT_VERSION

class HFClassifier:
    def __init__(self,model_path,adapter_path=None):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM,AutoTokenizer
        if int(transformers.__version__.split('.')[0]) != 4:
            raise RuntimeError('Use the isolated notebook environment with Transformers 4; do not upgrade between training and evaluation')
        path=Path(model_path).resolve()
        if not (path/"config.json").is_file():
            raise ValueError("Existing local HF checkpoint required")
        tokenizer_path = str(Path(adapter_path).resolve()) if adapter_path else str(path)
        self.tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
        if not self.tokenizer.chat_template:
            raise ValueError('Saved tokenizer is missing its training chat template')
        self.model=AutoModelForCausalLM.from_pretrained(str(path),local_files_only=True,device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,attn_implementation="eager")
        if adapter_path:
            from peft import PeftModel
            self.model=PeftModel.from_pretrained(self.model,str(Path(adapter_path).resolve()),local_files_only=True)
        self.model.eval(); self.identity=path.name + ('+adapter' if adapter_path else '')
    def predict(self,text):
        import torch
        if torch.cuda.is_available(): torch.cuda.synchronize()
        started=time.perf_counter()
        inputs=self.tokenizer.apply_chat_template(messages(text),tokenize=True,add_generation_prompt=True,
            return_tensors="pt",return_dict=True)
        ids=inputs['input_ids']
        if ids.shape[-1]>1000:
            raise ValueError("Ticket exceeds 1000-token limit")
        inputs={key:value.to(self.model.device) for key,value in inputs.items()}
        with torch.inference_mode():
            out=self.model.generate(**inputs,max_new_tokens=24,do_sample=False,use_cache=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                return_dict_in_generate=False)
        raw=self.tokenizer.decode(out[0,ids.shape[-1]:],skip_special_tokens=True).strip()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        return {**decision(raw),"raw_output":raw,"model_id":self.identity,"prompt_version":PROMPT_VERSION,
                "latency_ms":round((time.perf_counter()-started)*1000,2)}
