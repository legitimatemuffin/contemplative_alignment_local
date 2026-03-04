# local_model_backend.py

import torch, time
from typing import Tuple, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class LocalModelBackend:
    """Unified local model interface for AILuminate-compatible SUTs."""

    def __init__(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
                 quantize: bool = True,
                 cache_dir: str = "/local/scratch/a/asprigle/hf_cache"):
        self.model_id = model_id
        self.cache_dir = cache_dir

        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, cache_dir=cache_dir)

        bnb_config = None
        if quantize:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            trust_remote_code=True,
            cache_dir=cache_dir
        )

        self.device = next(self.model.parameters()).device
        print(f"✅ Local model loaded on {self.device}: {model_id}")

    def generate(self, prompt: str, temperature: float = 0.7, max_new_tokens: int = 500
                 ) -> Tuple[str, float, Dict[str, Any]]:
        start = time.time()
        formatted = f"[INST] {prompt.strip()} [/INST]" if "[INST]" not in prompt else prompt

        inputs = self.tok(formatted, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                do_sample=True
            )
        response = self.tok.decode(outputs[0], skip_special_tokens=True)
        elapsed = time.time() - start
        metadata = {
            "model_id": self.model_id,
            "quantized": True,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens
        }
        return response, elapsed, metadata
