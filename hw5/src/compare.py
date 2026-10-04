"""Базовая модель против адаптера на пяти фиксированных промптах.

Генерация жадная (do_sample=False): иначе сравнение «до/после» зависит от
удачи, а проверка «адаптер на чистой машине отвечает так же» теряет смысл.
Результат: docs/compare.md и metrics/compare_<variant>.json.
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import load_params
from src.runtime import resolve_device, resolve_dtype


def load_adapter_tokenizer(adapter_dir: Path):
    return AutoTokenizer.from_pretrained(load_params()["model"]["name"])


@torch.no_grad()
def generate(model, tok, prompts: list[str], system: str, params: dict, device) -> list[str]:
    out = []
    for user in prompts:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=params["model"].get("enable_thinking", False),
        )
        ids = tok(text, return_tensors="pt").to(device)
        gen = model.generate(**ids, max_new_tokens=params["compare"]["max_new_tokens"],
                             do_sample=False, pad_token_id=tok.pad_token_id)
        out.append(tok.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all_layers")
    ap.add_argument("--adapter-dir", default=None)
    ap.add_argument("--only-adapter", action="store_true", help="не генерировать базовой моделью")
    ap.add_argument("--out", default=None, help="куда писать json (проверка на чистой машине)")
    args = ap.parse_args()

    params = load_params()
    device = resolve_device(params["model"]["device"])
    dtype = resolve_dtype(params["model"]["dtype"])
    adapter_dir = Path(args.adapter_dir or Path(params["paths"]["models"]) / f"adapter_{args.variant}")
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    tok = load_adapter_tokenizer(adapter_dir)
    prompts, system = params["compare"]["prompts"], params["compare"]["system"]

    base = AutoModelForCausalLM.from_pretrained(cfg["base_model_name_or_path"], dtype=dtype).to(device)
    base.eval()
    before = [] if args.only_adapter else generate(base, tok, prompts, system, params, device)
    model = PeftModel.from_pretrained(base, adapter_dir).to(device)
    model.eval()
    after = generate(model, tok, prompts, system, params, device)

    result = {"variant": args.variant, "adapter_dir": str(adapter_dir), "base": before, "adapter": after}
    out = Path(args.out) if args.out else Path(params["paths"]["metrics"]) / f"compare_{args.variant}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.out:
        print(f"-> {out}")
        return

    lines = ["# Базовая модель против адаптера", "",
             f"Адаптер: `{adapter_dir}`. Генерация жадная, до {params['compare']['max_new_tokens']} токенов.", ""]
    for i, (p, b, a) in enumerate(zip(prompts, before, after), 1):
        q = p.split("Вопрос:\n")[-1].split("\n\nВарианты")[0]
        lines += [f"## {i}. {q}", "", "**База:**", "", f"> {b.replace(chr(10), ' / ')}", "",
                  "**Адаптер:**", "", f"> {a.replace(chr(10), ' / ')}", ""]
    Path(params["paths"]["compare"]).write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {out}, {params['paths']['compare']}")
    for b, a in zip(before, after):
        print(f"  база:    {b[:70]!r}\n  адаптер: {a[:70]!r}")


if __name__ == "__main__":
    main()
