#!/usr/bin/env bash
# Самопроверка домашней работы 2.
# Зелёный check.sh необходим для сдачи, но не достаточен: код читается глазами.
set -uo pipefail
cd "$(dirname "$0")/.."

fails=0
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$1"; fails=$((fails+1)); }
log=$(mktemp)
probe=$(mktemp)      # сюда ляжет JSON одного замера из проверки 5
trap 'rm -f "$log" "$probe"' EXIT

echo
echo "1. Сумма параметров по таблице сходится с моделью"
if uv run python - > "$log" 2>&1 <<'PY'
from src.config import load_params
from src.inspect_model import group_table, parameter_rows, resolve_device
from src.model import load_model

params = load_params()
params["model"]["device"] = str(resolve_device(params))
_, model = load_model(params)

by_table = sum(item["params"] for item in group_table(parameter_rows(model)))
direct = sum(p.numel() for p in model.parameters())
assert by_table == direct, (
    f"по таблице {by_table:,}, а в модели {direct:,} — "
    f"похоже, tied embeddings посчитаны дважды (+{by_table - direct:,})"
)
print(f"параметров: {direct:,}")
PY
then
  ok "$(tail -1 "$log")"
else
  fail "сумма по таблице не сходится с sum(p.numel() for p in model.parameters())"
  grep -E "AssertionError|Error" "$log" | tail -2 | sed 's/^/      /'
fi

echo
echo "2. Расчётное число LoRA-параметров совпадает с peft"
if uv run python - > "$log" 2>&1 <<'PY'
from src.config import load_params
from src.inspect_model import lora_report, resolve_device
from src.model import load_model

params = load_params()
params["model"]["device"] = str(resolve_device(params))
_, model = load_model(params)

for item in lora_report(model, params):
    assert item["match"], (
        f"{item['name']}: формула даёт {item['formula']:,}, "
        f"peft — {item['peft']:,}"
    )
    print(f"{item['name']}: {item['peft']:,}", end="; ")
PY
then
  ok "$(tail -1 "$log")"
else
  fail "своя формула расходится с print_trainable_parameters()"
  grep -E "AssertionError|Error" "$log" | tail -2 | sed 's/^/      /'
fi

echo
echo "3. Хуки снимаются: повторный прогон не копит их на модулях"
if uv run python - > "$log" 2>&1 <<'PY'
from src.config import load_params
from src.inspect_model import activation_norms, resolve_device
from src.model import load_model

params = load_params()
params["model"]["device"] = str(resolve_device(params))
tokenizer, model = load_model(params)


def hooks_left() -> int:
    return sum(len(m._forward_hooks) for m in model.modules())


activation_norms(tokenizer, model, params)
after_first = hooks_left()
activation_norms(tokenizer, model, params)
after_second = hooks_left()

assert after_first == 0, f"после первого прогона на модулях осталось хуков: {after_first}"
assert after_second == after_first, (
    f"хуки копятся: {after_first} после первого прогона, {after_second} после второго"
)
print("после двух прогонов зарегистрированных хуков: 0")
PY
then
  ok "$(tail -1 "$log")"
else
  fail "хуки не снимаются — handle.remove() отсутствует"
  grep -E "AssertionError|Error" "$log" | tail -2 | sed 's/^/      /'
fi

echo
echo "4. Три режима памяти дают разные числа (full FT > LoRA > инференс)"
if uv run python - > "$log" 2>&1 <<'PY'
from src.config import load_params
from src.inspect_model import memory_profile

peak = {item["mode"]: item["peak_mb"] for item in memory_profile(load_params())}
print(" ".join(f"{mode}={value:.0f}МБ" for mode, value in peak.items()))
assert peak["full_ft"] > peak["lora"], (
    f"full FT ({peak['full_ft']:.0f} МБ) должен быть дороже LoRA "
    f"({peak['lora']:.0f} МБ) — это не пик, а снимок"
)
assert peak["lora"] > peak["inference"], (
    f"LoRA ({peak['lora']:.0f} МБ) должна быть дороже инференса "
    f"({peak['inference']:.0f} МБ): активации для backward никуда не делись"
)
PY
then
  ok "$(head -1 "$log")"
else
  fail "режимы не различаются по памяти — меряется снимок в конце, а не пик"
  grep -E "МБ|AssertionError" "$log" | tail -3 | sed 's/^/      /'
fi

echo
echo "5. Проект запускается модулем: python -m src.inspect_model --probe inference"
# Проверяется поведение, а не текст: код возврата и непустой вывод.
# Плоские импорты (from config import ... вместо from src.config import ...)
# живут при запуске файлом и падают при запуске модулем — здесь они и видны.
uv run python -m src.inspect_model --probe inference > "$probe" 2> "$log"; rc=$?
if [ "$rc" -ne 0 ]; then
  fail "запуск модулем завершился с кодом $rc — проверьте импорты (from src.… ) и entry point"
  grep -E "Error|error:" "$log" | tail -3 | sed 's/^/      /'
elif [ ! -s "$probe" ]; then
  fail "запуск прошёл с кодом 0, но не напечатал ни строки — замер ничего не вернул"
else
  ok "код возврата 0, замер напечатал $(wc -l < "$probe" | tr -d ' ') строк(и)"
fi

echo
echo "6. В замере названа метрика памяти (и на ускорителе это не RSS)"
if [ ! -s "$probe" ]; then
  fail "нет вывода замера из проверки 5 — разбирать нечего"
elif uv run python - "$probe" > "$log" 2>&1 <<'PY'
import json
import sys

lines = [line for line in open(sys.argv[1], encoding="utf-8").read().splitlines() if line.strip()]
assert lines, "замер не напечатал ни строки"
data = json.loads(lines[-1])

source = str(data.get("metric_source") or "")
device = str(data.get("device", ""))
peak = float(data["peak_mb"])
assert source, "в записи замера нет непустого metric_source: из отчёта не видно, чем мерили"
assert peak > 0, f"peak_mb = {peak}: нулевой пик — это не замер"
if device.startswith(("mps", "cuda")):
    assert source not in ("ru_maxrss", "peak_wset"), (
        f"на {device} память измерена через {source}, то есть по процессу: тензоры лежат "
        "в памяти ускорителя и в RSS почти не попадают — режимы будут неразличимы"
    )
    assert source.startswith("torch."), f"на {device} ждём метрику ускорителя, а не {source}"
print(f"{device}: {source}, пик {peak:.0f} МБ")
PY
then
  ok "$(tail -1 "$log")"
else
  fail "метрика памяти не названа либо на ускорителе меряется RSS процесса"
  grep -E "AssertionError|Error" "$log" | tail -2 | sed 's/^/      /'
fi

echo
echo "7. В docs/anatomy.md записаны условия замера"
if uv run python - > "$log" 2>&1 <<'PY'
import re
from pathlib import Path

from src.config import load_params
from src.inspect_model import resolve_device

path = Path("docs/anatomy.md")
assert path.exists(), "нет docs/anatomy.md — сначала make inspect"
text = path.read_text(encoding="utf-8")
assert text.strip(), "docs/anatomy.md пуст"

params = load_params()
device = str(resolve_device(params))
dtype = params["model"]["dtype"]

required = {
    "раздел «Условия замера»": r"Условия замера",
    "платформа и версия ОС": r"(?i)(platform|платформа)[^\n]*\d",
    f"устройство {device}": rf"\b{re.escape(device)}\b",
    f"dtype {dtype}": re.escape(dtype),
    "версия torch": r"torch[^\n]*\d+\.\d+",
    "версия transformers": r"transformers[^\n]*\d+\.\d+",
    "чем измерена память": r"(driver_allocated_memory|max_memory_allocated|ru_maxrss|peak_wset)",
    "число прогонов": r"(?i)прогонов",
}
missing = [name for name, pattern in required.items() if not re.search(pattern, text)]
assert not missing, "в отчёте не указано: " + "; ".join(missing)
print(f"условия замера на месте ({len(required)} пунктов), отчёт совпадает с params.yaml")
PY
then
  ok "$(tail -1 "$log")"
else
  fail "в docs/anatomy.md нет условий замера или они не совпадают с params.yaml"
  grep -E "AssertionError|Error" "$log" | tail -2 | sed 's/^/      /'
fi

echo
if [ "$fails" -eq 0 ]; then
  printf '\033[32mВсе проверки пройдены.\033[0m Не забудьте docs/anatomy.md с таблицами и графиком.\n\n'
else
  printf '\033[31mПровалено проверок: %s\033[0m\n\n' "$fails"
  exit 1
fi
