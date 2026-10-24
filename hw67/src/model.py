"""MLflow models-from-code: адаптер + токенизатор + закреплённая база."""
import json
import os
from pathlib import Path

import mlflow


class AdapterModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
        os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        root = Path(context.artifacts["adapter"])
        self.manifest = json.loads((root / "manifest.json").read_text())
        cfg = self.manifest["model"]
        device = os.environ.get("HW6_DEVICE", cfg["device"])
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = getattr(torch, os.environ.get("HW6_DTYPE", cfg["dtype"]))
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(cfg["name"], revision=cfg["revision"], dtype=dtype)
        self.model = PeftModel.from_pretrained(base, root).to(device).eval()
        self.device = device

    def predict(self, context, model_input, params=None):
        import torch
        outputs = []
        g = self.manifest["generation"]
        for prompt in model_input["prompt"]:
            messages = [{"role": "system", "content": g["system"]}, {"role": "user", "content": str(prompt)}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.manifest["model"]["enable_thinking"])
            batch = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                out = self.model.generate(**batch, max_new_tokens=g["max_new_tokens"], do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
            outputs.append(self.tokenizer.decode(out[0, batch["input_ids"].shape[1]:], skip_special_tokens=True))
        return outputs


mlflow.models.set_model(AdapterModel())
