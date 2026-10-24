"""Короткий LoRA-ран из ДЗ 5 с логированием на каждом шаге оптимизатора."""
import gc
import math
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import save_json
from src.data import batches, load_split
from src.runtime import allocated_bytes, memory_metric, resolve_device, resolve_dtype, set_seed


@torch.no_grad()
def evaluate(model, examples, pad_id, device):
    model.eval()
    total, tokens = 0.0, 0
    for batch in batches(examples, 1, pad_id, False, 0):
        batch = {k: v.to(device) for k, v in batch.items()}
        n = int((batch["labels"][:, 1:] != -100).sum())
        if n:
            total += float(model(**batch).loss) * n
            tokens += n
    model.train()
    if tokens == 0:
        raise ValueError("В val нет токенов ответа под loss")
    value = total / tokens
    if not math.isfinite(value):
        raise ValueError("val loss не конечен")
    return value


def train(params, candidate, out, log):
    t, m = params["train"], params["model"]
    set_seed(t["seed"])
    device, dtype = resolve_device(m["device"]), resolve_dtype(m["dtype"])
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    tr, va = (load_split(params["data"][s]) for s in ("train", "val"))
    for b in (tr, va):
        if b.get("model") != m["name"]:
            raise ValueError("Тензоры токенизированы другой моделью: повторите ДЗ 4 с выбранной моделью")
    examples = tr["examples"][:t["train_limit"]]
    val = va["examples"][:t["val_limit"]]
    if tr["pad_token_id"] != va["pad_token_id"]:
        raise ValueError("pad_token_id train и val различаются")
    tok = AutoTokenizer.from_pretrained(m["name"], revision=m["revision"])
    model = AutoModelForCausalLM.from_pretrained(m["name"], revision=m["revision"], dtype=dtype).to(device)
    revision = getattr(model.config, "_commit_hash", None)
    if not revision:
        raise ValueError("Не удалось закрепить commit базовой модели Hugging Face")
    if t["gradient_checkpointing"]:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    cfg = LoraConfig(r=candidate["r"], lora_alpha=candidate["r"] * params["lora"]["alpha_over_r"],
        lora_dropout=params["lora"]["dropout"], target_modules=candidate["target_modules"], task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=candidate["lr"])
    pad = tr["pad_token_id"]
    start = time.perf_counter()
    base = evaluate(model, val, pad, device)
    log({"val_loss": base}, 0)
    peak, step, epoch = allocated_bytes(device), 0, 0
    while step < t["max_steps"]:
        stream = list(batches(examples, t["batch_size"], pad, True, t["seed"] + epoch))
        for pos in range(0, len(stream), t["grad_accum"]):
            group = stream[pos:pos + t["grad_accum"]]
            # Последняя неполная группа тоже даёт шаг; нормируем на токены ответа.
            counts = [int((b["labels"][:, 1:] != -100).sum()) for b in group]
            n_tokens = sum(counts)
            if not n_tokens:
                raise ValueError("В train-батче нет токенов ответа")
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for batch, count in zip(group, counts):
                if not count:
                    continue
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss * (count / n_tokens)
                if not torch.isfinite(loss):
                    raise ValueError("train loss не конечен")
                loss.backward()
                loss_sum += float(loss.detach())
                peak = max(peak, allocated_bytes(device))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            metrics = {"train_loss": loss_sum}
            if step % t["eval_every"] == 0 or step == t["max_steps"]:
                metrics["val_loss"] = evaluate(model, val, pad, device)
                print(f"{candidate['name']} step={step} val={metrics['val_loss']:.4f}", flush=True)
            log(metrics, step)
            if step >= t["max_steps"]:
                break
        epoch += 1
    seconds = time.perf_counter() - start
    out = Path(out)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    manifest = {"model": {**m, "revision": revision}, "generation": params["generation"], "candidate": candidate}
    save_json(out / "manifest.json", manifest)
    # Генерация тем же кодом, который затем загрузится из реестра.
    from src.model import AdapterModel
    import pandas as pd
    wrapper = AdapterModel()
    wrapper.model, wrapper.tokenizer, wrapper.device, wrapper.manifest = model.eval(), tok, device, manifest
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    prompts = params["generation"]["prompts"]
    answers = wrapper.predict(None, pd.DataFrame({"prompt": prompts}))
    results = {"base_val_loss": base, "final_val_loss": metrics["val_loss"],
        "seconds": seconds, "steps": step, "peak_memory_mb": peak / 1048576,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "adapter_mb": sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1048576}
    details = {"base_revision": revision, "device": str(device), "memory_metric": memory_metric(device),
        "train_rows": len(examples), "val_rows": len(val)}
    del wrapper, model, optimizer
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return results, details, answers
