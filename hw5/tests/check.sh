#!/usr/bin/env bash
# Самопроверка домашней работы 5.
# Полное обучение здесь НЕ запускается — оно идёт минуты. Проверяются
# артефакты последнего `make train && make compare` и то, что они сделаны
# текущей версией кода. Плюс два коротких прогона на воспроизводимость.
set -uo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")/.."
export HF_HUB_VERBOSITY=error TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false

fails=0
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$1"; fails=$((fails+1)); }
why()  { printf '%s\n' "$1" | sed 's/^/      /'; }
# Песочница для копии адаптера и smoke-прогонов. Прошлые песочницы, брошенные
# прерванной проверкой (kill -9 и закрытый терминал не вызывают trap), — удаляем.
rm -rf "${TMPDIR:-/tmp}"/hw5check.* 2>/dev/null
SANDBOX=$(mktemp -d "${TMPDIR:-/tmp}/hw5check.XXXXXX")
trap 'rm -rf "$SANDBOX"' EXIT INT TERM

# Общий помощник: метрики варианта + свежесть относительно текущего кода.
py_metrics() {
uv run python - "$1" <<'PY'
import json, sys
from pathlib import Path
from src.config import load_params
from src.train import inputs_fingerprint
variant = sys.argv[1]
params = load_params()
p = Path(params["paths"]["metrics"]) / f"train_{variant}.json"
if not p.exists():
    sys.exit(f"нет {p} — сначала make train")
m = json.loads(p.read_text(encoding="utf-8"))
now = inputs_fingerprint(params)
if m.get("inputs_fingerprint") != now:
    sys.exit(f"{p} сделан другой версией кода или конфига ({m.get('inputs_fingerprint')} ≠ {now}) — "
             "перезапустите make train")
print(json.dumps(m, ensure_ascii=False))
PY
}

# Первый контакт: обучения ещё не было — проверять нечего. Одна понятная
# причина вместо девяти трейсбеков.
if [ ! -f metrics/train_all_layers.json ]; then
  echo
  echo "Нет metrics/train_all_layers.json — обучения ещё не было, проверять нечего."
  missing=$(uv run python -c "
from pathlib import Path
from src.config import load_params
d = load_params()['data']
print(' '.join(p for p in (d['train'], d['val']) if not Path(p).exists()))
" 2>/dev/null)
  if [ -n "$missing" ]; then
    echo "Нет и входа: $missing"
    echo "Это выход стадии tokenize из ДЗ 4 — скопируйте свои train.pt и val.pt в data/tokenized/."
  fi
  echo "Порядок: make train && make compare && make check"
  if command -v shasum >/dev/null 2>&1; then fp=$(shasum -a 256 "$SELF" | cut -c1-12); else fp=$(sha256sum "$SELF" | cut -c1-12); fi
  printf '\nотпечаток tests/check.sh: \033[35m%s\033[0m\n\n' "$fp"
  exit 1
fi

echo
echo "1. Валидация считается"
if M=$(py_metrics all_layers 2>&1) && out=$(uv run python -c "
import json, sys
m = json.loads(sys.argv[1])
v = m.get('curve_val') or []
if len(v) < 3:
    sys.exit(f'точек val loss: {len(v)} — нужен замер до обучения, по ходу и в конце')
if v[0][0] != 0:
    sys.exit('нет замера на шаге 0 — не с чем сравнивать базовую модель')
print(f'точек val loss: {len(v)}, до обучения {v[0][1]}, в конце {v[-1][1]}')
" "$M" 2>&1); then
  ok "$out"
else
  fail "val loss не считается либо артефакты устарели — обучение вслепую"
  why "${out:-$M}"
fi

echo
echo "2. Обучение сошлось, адаптер лучше базовой модели"
if out=$(uv run python -c "
import json, math, sys
m = json.loads(sys.argv[1])
t = [x[1] for x in m['curve_train']]
if m.get('diverged') or not all(math.isfinite(x) for x in t):
    sys.exit('лосс ушёл в nan/inf — обучение разошлось')
k = max(1, len(t) // 5)
head, tail = sum(t[:k]) / k, sum(t[-k:]) / k
if tail >= head:
    sys.exit(f'train loss не падает: начало {head:.3f}, конец {tail:.3f}')
if m.get('final_val_loss') is None or m.get('base_val_loss') is None:
    sys.exit('val loss не считался — сравнить адаптер с базовой моделью нечем')
if m['final_val_loss'] >= m['base_val_loss']:
    sys.exit(f'val loss не лучше базовой модели: {m[\"final_val_loss\"]} против {m[\"base_val_loss\"]}')
print(f'train {head:.3f} → {tail:.3f}, val {m[\"base_val_loss\"]} → {m[\"final_val_loss\"]}')
" "$M" 2>&1); then
  ok "$out"
else
  fail "обучение не сошлось"
  why "$out"
fi

echo
echo "3. Адаптер весит мегабайты, а не сотни мегабайт"
if out=$(uv run python -c "
import json, sys
from pathlib import Path
m = json.loads(sys.argv[1])
cfg = json.loads((Path(m['adapter_dir']) / 'adapter_config.json').read_text(encoding='utf-8'))
full = cfg.get('modules_to_save') or []
bad = [t for t in (cfg.get('target_modules') or []) if any(x in str(t) for x in ('embed', 'lm_head'))]
if full:
    sys.exit(f'modules_to_save = {full}: эти модули уезжают в адаптер целиком, а не низкоранговой поправкой')
if bad:
    sys.exit(f'LoRA на {bad}: 155 млн параметров словаря под адаптером')
if m['adapter_size_mb'] > 100:
    sys.exit(f'адаптер {m[\"adapter_size_mb\"]} МБ — для r={cfg[\"r\"]} это слишком')
print(f'{m[\"adapter_size_mb\"]} МБ, r={cfg[\"r\"]}, обучаемых {m[\"trainable_share\"]:.2%}')
" "$M" 2>&1); then
  ok "$out"
else
  fail "адаптер тяжёлый — в него попало то, что не должно обучаться"
  why "$out"
fi

echo
echo "4. Адаптер поднимается на чистой машине и отвечает так же"
REF=metrics/compare_all_layers.json
if [ ! -s "$REF" ]; then
  fail "нет $REF — сначала make compare"
else
  ADIR=$(uv run python -c "import json; print(json.load(open('$REF'))['adapter_dir'])")
  cp -R "$ADIR" "$SANDBOX/adapter"
  # Чистая машина: только папка адаптера, без сети. Базовая модель — из кеша HF.
  # Чистая машина: всё, кроме весов базовой модели, берётся из папки адаптера.
  # Загрузка своя, не через src/compare.py — проверка не должна зависеть от кода, который проверяет.
  if HF_HUB_OFFLINE=1 uv run python - "$SANDBOX/adapter" "$SANDBOX/again.json" > "$SANDBOX/cmp.log" 2>&1 <<'PY'
import json, sys
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.compare import generate
from src.config import load_params
from src.runtime import resolve_device, resolve_dtype
adir = Path(sys.argv[1])
if not (adir / "tokenizer_config.json").exists():
    sys.exit("в папке адаптера нет токенизатора — на чистой машине не с чем его поднять")
params = load_params()
device, dtype = resolve_device(params["model"]["device"]), resolve_dtype(params["model"]["dtype"])
tok = AutoTokenizer.from_pretrained(adir)
base_name = json.loads((adir / "adapter_config.json").read_text())["base_model_name_or_path"]
base = AutoModelForCausalLM.from_pretrained(base_name, dtype=dtype).to(device)
model = PeftModel.from_pretrained(base, adir).to(device).eval()
out = generate(model, tok, params["compare"]["prompts"], params["compare"]["system"], params, device)
Path(sys.argv[2]).write_text(json.dumps({"adapter": out}, ensure_ascii=False))
PY
  then
    if out=$(uv run python -c "
import json, sys
a = json.load(open(sys.argv[1]))['adapter']; b = json.load(open(sys.argv[2]))['adapter']
diff = [i + 1 for i, (x, y) in enumerate(zip(a, b)) if x != y]
if diff:
    sys.exit(f'ответы разошлись на промптах {diff}: {a[diff[0]-1][:60]!r} vs {b[diff[0]-1][:60]!r}')
print(f'{len(a)} ответов совпали дословно')
" "$REF" "$SANDBOX/again.json" 2>&1); then
      ok "$out"
    else
      fail "адаптер с чистой машины отвечает иначе"
      why "$out"
    fi
  else
    fail "адаптер не поднимается из своей папки"
    why "$(tail -3 "$SANDBOX/cmp.log")"
  fi
fi
rm -rf "$SANDBOX/adapter"   # копия адаптера больше не нужна

echo
echo "5. Два прогона с одним конфигом дают одну кривую"
for i in 1 2; do
  uv run python -m src.train --variant all_layers --max-steps 3 --val-limit 8 \
     --out "$SANDBOX/run$i" > "$SANDBOX/run$i.log" 2>&1
done
if out=$(uv run python -c "
import json, sys
a = json.load(open(sys.argv[1] + '/metrics/train_all_layers.json'))['curve_train']
b = json.load(open(sys.argv[2] + '/metrics/train_all_layers.json'))['curve_train']
if a != b:
    sys.exit(f'кривые разные: {[x[1] for x in a]} vs {[x[1] for x in b]} — сид не зафиксирован')
print(f'train loss по шагам: {[x[1] for x in a]} — одинаково')
" "$SANDBOX/run1" "$SANDBOX/run2" 2>&1); then
  ok "$out"
else
  fail "прогоны невоспроизводимы — сравнивать эксперименты нельзя"
  why "$out"
fi
rm -rf "$SANDBOX"/run1/adapter_* "$SANDBOX"/run2/adapter_*   # нужны были только кривые

echo
echo "6. Эксперимент с заморозкой первых слоёв проведён"
if F=$(py_metrics freeze14 2>&1) && out=$(uv run python -c "
import json, sys
from pathlib import Path
a, f = json.loads(sys.argv[1]), json.loads(sys.argv[2])
cfg = json.loads((Path(f['adapter_dir']) / 'adapter_config.json').read_text(encoding='utf-8'))
layers = cfg.get('layers_to_transform') or []
if f['freeze_first'] <= 0 or not layers or min(layers) != f['freeze_first']:
    sys.exit('у варианта freeze14 адаптеры висят и на нижних слоях — заморозки нет')
if f['trainable_params'] >= a['trainable_params']:
    sys.exit('обучаемых параметров не меньше, чем у полного варианта')
print(f\"обучаемых {f['trainable_params']:,} против {a['trainable_params']:,}; \"
      f\"{f['seconds']:.0f} с против {a['seconds']:.0f} с; val {f['final_val_loss']} против {a['final_val_loss']}\")
" "$M" "$F" 2>&1); then
  ok "$out"
else
  fail "второго варианта нет либо заморозка не сработала"
  why "${out:-$F}"
fi

echo
echo "7. Стадии train и compare объявлены в dvc.yaml"
if out=$(uv run python - <<'PY' 2>&1
import sys, yaml
d = yaml.safe_load(open("dvc.yaml", encoding="utf-8")) or {}
st = d.get("stages") or {}
need = {"train_all": "models/adapter_all_layers", "train_freeze": "models/adapter_freeze14"}
for name, out in need.items():
    s = st.get(name)
    if not s:
        sys.exit(f"нет стадии {name}")
    if "src/train.py" not in (s.get("deps") or []):
        sys.exit(f"{name}: src/train.py не в deps — правка кода не пересчитает адаптер")
    if out not in [str(x) for x in (s.get("outs") or [])]:
        sys.exit(f"{name}: {out} не в outs")
if "compare" not in st:
    sys.exit("нет стадии compare")
print("train_all, train_freeze, compare")
PY
); then
  ok "$out"
else
  fail "dvc.yaml не описывает обучение"
  why "$out"
fi

echo
echo "8. Разбор дефектов написан"
if [ ! -s docs/defects.md ]; then
  fail "нет docs/defects.md"
else
  heads=$(grep -cE '^#{1,3} ' docs/defects.md || true); words=$(wc -w < docs/defects.md | tr -d ' ')
  if [ "$heads" -lt 5 ] || [ "$words" -lt 250 ]; then
    fail "docs/defects.md слишком короткий: разделов $heads, слов $words — по разделу на дефект, с числами"
  else
    ok "docs/defects.md: разделов $heads, слов $words"
  fi
fi

echo
echo "9. Гигиена репозитория"
junk=$(git ls-files 2>/dev/null | grep -E '(^|/)(\.DS_Store|__pycache__|\.venv|.*\.bak|.*\.orig|.*\.safetensors|.*\.pt|.*\.bin)$|^models/|^data/' || true)
big=$(git ls-files -z 2>/dev/null | xargs -0 -I{} sh -c 'test -f "{}" && s=$(wc -c < "{}") && [ "$s" -gt 5242880 ] && echo "{} ($((s/1048576)) МБ)"' 2>/dev/null || true)
if [ -z "$junk" ] && [ -z "$big" ]; then
  ok "в git нет весов, данных, мусора и файлов тяжелее 5 МБ"
else
  fail "в git попало лишнее"
  [ -n "$junk" ] && echo "$junk" | sed 's/^/      /'
  [ -n "$big" ] && echo "$big" | sed 's/^/      тяжёлый файл: /'
fi

echo
if command -v shasum >/dev/null 2>&1; then fp=$(shasum -a 256 "$SELF" | cut -c1-12); else fp=$(sha256sum "$SELF" | cut -c1-12); fi
printf 'отпечаток tests/check.sh: \033[35m%s\033[0m — на видео сдачи должен совпадать с выданным\n\n' "$fp"
if [ "$fails" -eq 0 ]; then
  printf '\033[32mВсе проверки пройдены.\033[0m Кривые: docs/curves.png, сравнение: docs/compare.md\n\n'
else
  printf '\033[31mПровалено проверок: %s\033[0m\n\n' "$fails"
  exit 1
fi
