"""Анатомия модели: параметры по слоям, нормы активаций, профиль памяти.

    python -m src.inspect_model            полный разбор, отчёт в docs/
    python -m src.inspect_model --probe M  один режим замера памяти (служебный
                                           вызов из отдельного процесса)

Файл называется inspect_model.py, а не inspect.py: имя inspect занято
модулем стандартной библиотеки, и его перекрытие ломает импорты в чужом коде.
"""

import argparse
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path

import peft
import torch
import transformers
from peft import LoraConfig, get_peft_model

from src.config import load_params
from src.model import build_prompt, load_model, set_seed

# Пик RSS снимается разными механизмами на разных ОС, поэтому оба импорта
# необязательные: resource есть на macOS и Linux, но его нет на Windows;
# psutil нужен на Windows, где peak_wset — единственный high-water mark,
# который отдаёт система. Код, написанный под одну ОС, у соседа не запустится.
try:
    import resource
except ImportError:
    resource = None

try:
    import psutil
except ImportError:
    psutil = None

# transformers читает safetensors в несколько потоков, и на связке
# pyo3 OnceLock + GIL загрузка иногда встаёт намертво: на этой машине
# примерно один процесс из шести не доживал до конца from_pretrained.
# Замер обязан быть воспроизводимым, поэтому читаем последовательно —
# на модели 0.6B это не стоит ничего (4.4 с против 4.5 с).
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

MODES = ("inference", "full_ft", "lora")

# Порядок задаёт порядок строк в таблице. Проверка идёт сверху вниз,
# поэтому «norm» стоит после проекций: в их именах слова norm нет.
GROUPS = (
    ("embed", ("embed_tokens",)),
    ("q_proj", ("q_proj",)),
    ("k_proj", ("k_proj",)),
    ("v_proj", ("v_proj",)),
    ("o_proj", ("o_proj",)),
    ("gate_proj", ("gate_proj",)),
    ("up_proj", ("up_proj",)),
    ("down_proj", ("down_proj",)),
    ("norm", ("norm",)),
    ("lm_head", ("lm_head",)),
)


def resolve_device(params: dict) -> torch.device:
    """Развернуть device: auto в конкретное устройство — ровно один раз.

    Строка «auto» уходит в device_map и включает диспетчер accelerate,
    который для шага обучения только мешает. Решаем здесь и передаём дальше
    уже конкретное имя.
    """
    name = params["model"]["device"]
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# 1. Параметры по типам модулей
# --------------------------------------------------------------------------

def group_of(name: str) -> str:
    """Тип модуля по имени параметра."""
    for group, marks in GROUPS:
        if any(mark in name for mark in marks):
            return group
    return "прочее"


def parameter_rows(model) -> list[dict]:
    """Все тензоры параметров модели.

    remove_duplicate=False — иначе в таблицу не попадёт lm_head.
    """
    rows = []
    for name, param in model.named_parameters(remove_duplicate=False):
        rows.append({
            "name": name,
            "shape": tuple(param.shape),
            "numel": param.numel(),
            "tied": False,
        })
    return rows


def group_table(rows: list[dict]) -> list[dict]:
    """Свод «тип модуля → shape → параметров → доля от всей модели».

    В params попадают только уникальные тензоры, в tied_params — то,
    что модуль переиспользует у соседа.
    """
    total = sum(r["numel"] for r in rows if not r["tied"])
    agg: dict[str, dict] = {}
    for row in rows:
        group = group_of(row["name"])
        item = agg.setdefault(group, {
            "group": group, "modules": 0, "shapes": [], "params": 0, "tied_params": 0,
        })
        item["modules"] += 1
        shape = "×".join(map(str, row["shape"]))
        if shape not in item["shapes"]:
            item["shapes"].append(shape)
        if row["tied"]:
            item["tied_params"] += row["numel"]
        else:
            item["params"] += row["numel"]

    order = [g for g, _ in GROUPS] + ["прочее"]
    table = [agg[g] for g in order if g in agg]
    for item in table:
        # В группе norm форм две (по голове и по hidden), показываем обе.
        item["shape"] = ", ".join(item.pop("shapes"))
        item["share"] = item["params"] / total
    return table


# --------------------------------------------------------------------------
# 2. Forward-hooks и нормы активаций
# --------------------------------------------------------------------------

def decoder_layers(model):
    """Список декодер-блоков. У Qwen3 это model.model.layers."""
    decoder = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    return decoder.layers


def hook_targets(model) -> dict[str, int]:
    """Первый, средний и последний блок — по номерам, а не по именам."""
    n_layers = len(decoder_layers(model))
    return {"первый": 0, "средний": n_layers // 2, "последний": n_layers - 1}


def forward_hooks(modules: dict) -> dict:
    """Навесить forward-hooks на модули и вернуть словарь, куда они пишут."""
    store: dict[str, list[float]] = {}

    def make_hook(label: str):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            store[label] = hidden[0].float().norm(dim=-1).detach().cpu().tolist()
        return hook

    for label, module in modules.items():
        module.register_forward_hook(make_hook(label))
    return store


def activation_norms(tokenizer, model, params: dict) -> dict:
    """L2-нормы скрытых состояний на выходе трёх блоков, по позициям токена."""
    layers = decoder_layers(model)
    targets = hook_targets(model)
    prompt = build_prompt(tokenizer, params, params["hooks"]["prompt"])
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    store = forward_hooks({label: layers[i] for label, i in targets.items()})
    with torch.inference_mode():
        model(**inputs)

    return {
        "layers": targets,
        "norms": {label: store[label] for label in targets},
        "n_tokens": inputs["input_ids"].shape[1],
    }


# --------------------------------------------------------------------------
# 3. Сколько параметров добавляет LoRA
# --------------------------------------------------------------------------

def lora_config(params: dict, cfg: dict) -> LoraConfig:
    """LoraConfig из params.yaml — ни r, ни target_modules в коде не зашиты."""
    return LoraConfig(
        r=cfg["r"],
        lora_alpha=params["lora"]["alpha_ratio"] * cfg["r"],
        lora_dropout=params["lora"]["dropout"],
        target_modules=list(cfg["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )


def lora_params_formula(model, r: int, target_modules) -> int:
    """Своя формула: на каждый целевой Linear ровно r * (in_features + out_features).

    A имеет форму (r, in), B — (out, r), смещений у них нет. Вся арифметика
    LoRA умещается в эту строчку, и она обязана сойтись с peft до штуки.
    """
    targets = set(target_modules)
    total = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in targets:
            total += r * (module.in_features + module.out_features)
    return total


def lora_report(model, params: dict) -> list[dict]:
    """Для каждого конфига: своя формула против peft.

    Адаптер снимается через unload(): дальше модель нужна чистой.
    """
    base_params = sum(p.numel() for p in model.parameters())
    result = []
    for cfg in params["lora"]["configs"]:
        expected = lora_params_formula(model, cfg["r"], cfg["target_modules"])

        peft_model = get_peft_model(model, lora_config(params, cfg))
        peft_model.print_trainable_parameters()
        trainable, total = peft_model.get_nb_trainable_parameters()
        model = peft_model.unload()
        model.requires_grad_(True)

        result.append({
            "name": cfg["name"],
            "r": cfg["r"],
            "target_modules": list(cfg["target_modules"]),
            "formula": expected,
            "peft": trainable,
            "match": expected == trainable,
            "total_with_adapter": total,
            "share_of_base": trainable / base_params,
        })
    return result


# --------------------------------------------------------------------------
# 4. Память в трёх режимах
# --------------------------------------------------------------------------

def device_allocated_bytes(device: torch.device) -> int:
    """Сколько памяти занято прямо сейчас."""
    used, _ = peak_rss()
    return used


def device_metric_source(device: torch.device) -> str:
    """Имя функции, которой снята память."""
    _, source = peak_rss()
    return source


def peak_rss() -> tuple[int, str]:
    """Пик RSS процесса в байтах И метка источника метрики.

    Метка возвращается не для красоты: «пик 1001 МБ» без указания, чем это
    снято, — не результат, а повод для спора. Тем более что RSS и память
    ускорителя — разные величины (см. PeakMemory ниже).

    Три ОС меряют по-разному:

    * macOS и Linux — `resource.getrusage(RUSAGE_SELF).ru_maxrss`, high-water
      mark процесса; на macOS он в байтах, на Linux в килобайтах;
    * Windows — `psutil.Process().memory_info().peak_wset`: модуля `resource`
      там нет вовсе. Обратное тоже верно — поля `peak_wset` нет на macOS и
      Linux, и код, написанный только под него, у соседа падает.

    Если недоступно ничего — исключение. Тихий ноль хуже отсутствия числа:
    ноль попадает в отчёт и его выдают за результат.
    """
    if resource is not None:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return (peak if sys.platform == "darwin" else peak * 1024), "ru_maxrss"
    if psutil is not None and hasattr(psutil.Process().memory_info(), "peak_wset"):
        return int(psutil.Process().memory_info().peak_wset), "peak_wset"
    raise RuntimeError(
        f"нечем снять пик RSS на платформе {sys.platform}: модуля resource нет, "
        "а psutil не установлен либо не отдаёт peak_wset. Выполните uv sync."
    )


class PeakMemory:
    """Сколько памяти занято к концу прогона."""

    def __init__(self, device: torch.device, interval: float = 0.01):
        self.device = device
        self.used = 0

    def __enter__(self) -> "PeakMemory":
        return self

    def __exit__(self, *exc) -> bool:
        # TODO: это расход режима — или то, что осталось занято после него,
        # когда всё уже посчитано и мусор собран?
        gc.collect()
        self.used = device_allocated_bytes(self.device)
        return False

    def result(self) -> dict:
        """Числа замера вместе с именем метрики, которой они сняты."""
        rss, rss_source = peak_rss()
        accelerator = self.device.type in ("mps", "cuda")
        return {
            "peak_mb": round((self.used if accelerator else rss) / 1024 ** 2, 1),
            "peak_device_mb": round(self.used / 1024 ** 2, 1),
            "peak_rss_mb": round(rss / 1024 ** 2, 1),
            "metric": (f"аллокатор {self.device.type}" if accelerator
                       else "RSS процесса"),
            "metric_source": (device_metric_source(self.device) if accelerator
                              else rss_source),
            "rss_source": rss_source,
        }


def measure_mode(mode: str, params: dict) -> dict:
    """Один режим: инференс / full fine-tune / LoRA.

    Обучение — ровно один шаг forward + backward + optimizer.step():
    пик памяти достигается уже на нём, гонять эпоху незачем.
    """
    device = resolve_device(params)
    params["model"]["device"] = str(device)
    set_seed(params["generate"]["seed"])

    started = time.perf_counter()
    loss = None

    _, model = load_model(params)

    with PeakMemory(device) as peak:
        ids = torch.randint(
            0, model.config.vocab_size,
            (params["memory"]["batch_size"], params["memory"]["seq_len"]),
            device=model.device,
        )
        if mode == "inference":
            model.eval()
            with torch.inference_mode():
                model(input_ids=ids)
        else:
            if mode == "lora":
                model = get_peft_model(model, lora_config(params, params["lora"]["configs"][0]))
            model.train()
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=float(params["memory"]["lr"]),
            )
            output = model(input_ids=ids, labels=ids)
            output.loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            loss = round(output.loss.detach().item(), 4)

    result = peak.result()
    result.update(
        mode=mode,
        device=str(device),
        seq_len=params["memory"]["seq_len"],
        batch_size=params["memory"]["batch_size"],
        seconds=round(time.perf_counter() - started, 1),
        loss=loss,
    )
    return result


def memory_profile(params: dict) -> list[dict]:
    """Профиль памяти в трёх режимах.

    memory.repeats задаёт число прогонов на режим; берётся худший (максимум).
    """
    repeats = max(1, int(params["memory"].get("repeats", 1)))
    results = []
    for mode in MODES:
        runs = [measure_mode(mode, params) for _ in range(repeats)]
        worst = max(runs, key=lambda item: item["peak_mb"])
        worst["repeats"] = repeats
        worst["peak_mb_runs"] = [item["peak_mb"] for item in runs]
        results.append(worst)
        gc.collect()
    return results


# --------------------------------------------------------------------------
# 5. Условия, без которых цифры замера ничего не значат
# --------------------------------------------------------------------------

def environment(params: dict, memory: list[dict]) -> dict:
    """Всё, что нужно, чтобы чужой замер можно было сравнить со своим.

    Расхождение в полтора раза между двумя машинами — норма, а не ошибка,
    но только если написано, чем эти машины отличались. Метрики памяти берутся
    из самих замеров, а не из предположений: что реально сработало в дочернем
    процессе, то и уходит в отчёт.
    """
    def unique(field: str) -> str:
        values = dict.fromkeys(str(item.get(field) or "") for item in memory)
        return ", ".join(value for value in values if value)

    return {
        "platform": platform.platform(),
        "system": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "device": params["model"]["device"],
        "dtype": params["model"]["dtype"],
        "seq_len": params["memory"]["seq_len"],
        "batch_size": params["memory"]["batch_size"],
        "repeats": max(1, int(params["memory"].get("repeats", 1))),
        "memory_metric": unique("metric_source"),
        "rss_metric": unique("rss_source"),
    }


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Разбор модели: параметры, активации, память")
    parser.add_argument("--probe", choices=MODES, help="служебный режим: замерить память и выйти")
    args = parser.parse_args()

    params = load_params()
    set_seed(params["generate"]["seed"])
    params["model"]["device"] = str(resolve_device(params))

    if args.probe:
        print(json.dumps(measure_mode(args.probe, params), ensure_ascii=False))
        return

    # Импорт здесь, а не наверху: matplotlib не нужен в служебных --probe
    # процессах, а тянется он заметно дольше остального.
    from src.report import write_report

    tokenizer, model = load_model(params)
    rows = parameter_rows(model)
    table = group_table(rows)
    total = sum(item["params"] for item in table)
    memory = memory_profile(params)

    report = {
        "model": params["model"]["name"],
        "dtype": params["model"]["dtype"],
        "device": params["model"]["device"],
        "environment": environment(params, memory),
        "config": {
            key: getattr(model.config, key)
            for key in ("num_hidden_layers", "hidden_size", "intermediate_size",
                        "num_attention_heads", "num_key_value_heads", "head_dim",
                        "vocab_size", "tie_word_embeddings")
        },
        "params_total": total,
        "params_direct": sum(p.numel() for p in model.parameters()),
        "params_by_group": table,
        "activations": activation_norms(tokenizer, model, params),
        "lora": lora_report(model, params),
        "memory": memory,
    }

    Path(params["report"]["json"]).parent.mkdir(exist_ok=True)
    Path(params["report"]["json"]).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(report, params)

    env = report["environment"]
    print(f"\nПараметров: {total:,} (по таблице) / {report['params_direct']:,} (напрямую)"
          .replace(",", " "))
    print(f"Условия: {env['platform']}, device {env['device']}, dtype {env['dtype']}, "
          f"seq_len {env['seq_len']}, прогонов на режим {env['repeats']}, "
          f"torch {env['torch']}, transformers {env['transformers']}")
    for mode in report["memory"]:
        print(f"  {mode['mode']:<10} пик {mode['peak_mb']:>8.1f} МБ  "
              f"({mode['metric']}: {mode['metric_source']})")
    print(f"\nОтчёт: {params['report']['markdown']}, график: {params['hooks']['plot']}")


if __name__ == "__main__":
    main()
