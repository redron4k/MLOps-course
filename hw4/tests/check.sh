#!/usr/bin/env bash
# Самопроверка домашней работы 4.
# Зелёный check.sh необходим для сдачи, но не достаточен: код читается глазами.
set -uo pipefail
# Отпечаток самого скрипта — печатается в конце. Правки check.sh не запрещены,
# но должны быть видны: на видео сдачи отпечаток сверяется с выданным.
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")/.."
# Служебные предупреждения HF Hub и transformers не относятся к проверкам
# и только засоряют вывод, который попадёт на видео.
export HF_HUB_VERBOSITY=error TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false

fails=0
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$1"; fails=$((fails+1)); }
why()  { printf '%s\n' "$1" | sed 's/^/      /'; }

# Проверяем то, что лежит на диске СЕЙЧАС, а не то, что осталось с прошлого прогона.
# Вход стадии — ваш датасет из ДЗ 3 в data/*.jsonl. Срез курса (make sample)
# собирается только если своих файлов нет: parquet для него есть у преподавателя.
TRAIN_JSONL=$(uv run python -c "from src.config import load_params; print(load_params()['data']['train_jsonl'])")
VAL_JSONL=$(uv run python -c "from src.config import load_params; print(load_params()['data']['val_jsonl'])")
if [ -s "$TRAIN_JSONL" ] && [ -s "$VAL_JSONL" ]; then
  echo "Вход стадии: $TRAIN_JSONL, $VAL_JSONL — make sample пропущен"
else
  echo "Входных JSONL нет — пробую собрать срез курса: make sample"
  if ! make sample > .check_sample.log 2>&1; then
    sed 's/^/  /' .check_sample.log | grep -v '^  make\|^  uv run' ; rm -f .check_sample.log
    exit 1
  fi
  rm -f .check_sample.log
fi
echo "Готовлю артефакты: make tokenize"
if ! make tokenize > .check_tokenize.log 2>&1; then
  echo "make tokenize упал:"; sed 's/^/  /' .check_tokenize.log | tail -20; rm -f .check_tokenize.log
  exit 1
fi
rm -f .check_tokenize.log

echo
echo "1. В лосс попадает только ответ ассистента"
if out=$(uv run python - <<'PY' 2>&1
import json, sys, torch
from transformers import AutoTokenizer
from src.config import load_params

params = load_params()
tok = AutoTokenizer.from_pretrained(params["model"]["name"])
blob = torch.load("data/tokenized/train.pt", weights_only=False)
answers = {}
for line in open(params["data"]["train_jsonl"], encoding="utf-8"):
    r = json.loads(line)
    answers[r["id"]] = r["messages"][-1]["content"]

max_len = params["tokenize"]["max_seq_len"]
checked = 0
for ex in blob["examples"]:
    ids = [i for i, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
    if not ids:
        sys.exit(f"{ex['id']}: в лосс не попал ни один токен")
    got = tok.decode(ids, skip_special_tokens=True).strip()
    want = answers[ex["id"]].strip()
    # Обрезанный по max_seq_len пример теряет хвост ответа — сверяем префикс.
    same = want.startswith(got) if len(ex["input_ids"]) >= max_len else got == want
    if not same:
        sys.exit(f"{ex['id']}: под маской не ответ ассистента\n"
                 f"ждали: {want[:100]!r}\nвышло: {got[:100]!r}")
    checked += 1
print(f"сверено примеров: {checked}")
PY
); then
  ok "декодированные позиции label != -100 — это ответ ассистента ($out)"
else
  fail "под маской лосса не только ответ: промпт не замаскирован либо граница уехала"
  why "$out"
fi

echo
echo "2. Шаблон обучения и шаблон инференса не разошлись"
if out=$(uv run python - <<'PY' 2>&1
import json, sys
from transformers import AutoTokenizer
from src.config import load_params
from src.prompt import build_chat_text

params = load_params()
tok = AutoTokenizer.from_pretrained(params["model"]["name"])
n = 0
for line in open(params["data"]["val_jsonl"], encoding="utf-8"):
    m = json.loads(line)["messages"]
    train_bytes = build_chat_text(tok, m, params, add_generation_prompt=False).encode("utf-8")
    infer_bytes = build_chat_text(tok, m, params, add_generation_prompt=True).encode("utf-8")
    if train_bytes[:len(infer_bytes)] != infer_bytes:
        i = next(k for k in range(min(len(train_bytes), len(infer_bytes)) + 1)
                 if train_bytes[k:k + 1] != infer_bytes[k:k + 1])
        cut = lambda b: b[max(0, i - 40):i + 60].decode("utf-8", "replace")
        sys.exit(f"разошлись на байте {i}\n"
                 f"inference: ...{cut(infer_bytes)}\n"
                 f"train:     ...{cut(train_bytes)}")
    n += 1
print(f"сверено диалогов: {n}")
PY
); then
  ok "строка inference-пути — побайтовый префикс строки train-пути ($out)"
else
  fail "train-путь и inference-путь дают разные строки: модель училась не на том формате"
  why "$out"
fi

echo
echo "3. Шаблон чата не собирается руками"
if grep -rqE '<\|im_(start|end)\|>' src/*.py 2>/dev/null; then
  fail "в src/ найдены спецтокены ролей строкой — шаблон обязан приходить из apply_chat_template"
  why "$(grep -nE '<\|im_(start|end)\|>' src/*.py)"
else
  ok "спецтокенов ролей в src/ нет"
fi
users=$(grep -l 'apply_chat_template' src/*.py 2>/dev/null | sort | tr '\n' ' ')
if [ "$users" = "src/prompt.py " ]; then
  ok "apply_chat_template вызывается ровно в одном месте: src/prompt.py"
else
  fail "apply_chat_template вызывается не только в src/prompt.py: ${users:-нигде}"
fi

echo
echo "4. Паддинг слева"
if out=$(uv run python - <<'PY' 2>&1
import json, sys, torch
from src.collate import DynamicPaddingCollator
from src.config import load_params

params = load_params()
side = params["tokenize"]["padding_side"]
if side != "left":
    sys.exit(f"tokenize.padding_side = {side!r}, для decoder-only нужен 'left'")

meta = json.load(open("metrics/tokenize.json", encoding="utf-8"))
if meta.get("padding_side") != "left":
    sys.exit(f"в metrics/tokenize.json padding_side = {meta.get('padding_side')!r}")

blob = torch.load("data/tokenized/train.pt", weights_only=False)
if blob.get("padding_side") != "left":
    sys.exit(f"в data/tokenized/train.pt padding_side = {blob.get('padding_side')!r}")

if DynamicPaddingCollator(blob["pad_token_id"]).padding_side != "left":
    sys.exit("у коллатора паддинг справа по умолчанию")

batch = sorted(blob["examples"][:16], key=lambda e: len(e["input_ids"]))[:2]
out = DynamicPaddingCollator(blob["pad_token_id"], side)(batch)
mask = out["attention_mask"][0].tolist()
if mask[0] != 0 or mask[-1] != 1:
    sys.exit(f"коллатор паддит не слева: attention_mask {mask[0]}...{mask[-1]}")
if out["labels"][0][0].item() != -100:
    sys.exit("labels на паддинге не -100 — модель учится предсказывать pad")
print("params + metrics + коллатор")
PY
); then
  ok "левый паддинг, labels на паддинге -100 ($out)"
else
  fail "паддинг справа: при batch > 1 генерация продолжает pad-токены, а не промпт"
  why "$out"
fi

echo
echo "5. Обрезка по max_seq_len посчитана и не превышает порог"
if out=$(uv run python - <<'PY' 2>&1
import json, sys
m = json.load(open("metrics/tokenize.json", encoding="utf-8"))
limit = m.get("truncated_warn_ratio")
if limit is None:
    sys.exit("в метриках нет truncated_warn_ratio — порог не задан")
parts = []
for name, s in m["splits"].items():
    if "truncated" not in s or "truncated_ratio" not in s:
        sys.exit(f"{name}: в метриках нет доли обрезанных примеров — обрезка молчаливая")
    if s["truncated_ratio"] > limit:
        sys.exit(f"{name}: обрезано {s['truncated_ratio']:.1%} при пороге {limit:.1%}")
    parts.append(f"{name} {s['truncated_ratio']:.1%}")
print(", ".join(parts) + f" при пороге {limit:.1%}")
PY
); then
  ok "доля обрезанных есть в metrics/tokenize.json и ниже порога ($out)"
else
  fail "обрезка молчаливая либо выше порога — часть ответов не доезжает до обучения"
  why "$out"
fi

echo
echo "6. Граница маски устойчива к склейке BPE"
if out=$(uv run python - <<'PY' 2>&1
import sys
from transformers import AutoTokenizer
from src.config import load_params
from src.prompt import prompt_token_len

params = load_params()
tok = AutoTokenizer.from_pretrained(params["model"]["name"])
# Промпт обрывается посреди слова: пары одинаковых токенов на границе нет,
# должен сработать запасной путь по символьным офсетам.
prompt_text, full_text = "Приве", "Привет, мир"
enc = tok(full_text, add_special_tokens=False, return_offsets_mapping=True)
n, fallback = prompt_token_len(tok, prompt_text, enc["input_ids"], enc["offset_mapping"])
if not fallback:
    sys.exit("запасной путь не сработал там, где BPE склеил границу")
if enc["offset_mapping"][n][0] < len(prompt_text):
    sys.exit("граница ответа попала внутрь промпта")
print(f"граница на токене {n}")
PY
); then
  ok "при склейке BPE граница берётся по символьным офсетам ($out)"
else
  fail "склейка BPE на границе промпт/ответ не обрабатывается"
  why "$out"
fi

echo
echo "7. Стадия tokenize объявлена в dvc.yaml и согласована с params.yaml"
if out=$(uv run python - <<'PY' 2>&1
import sys, yaml
from src.config import load_params
p = load_params()
try:
    d = yaml.safe_load(open("dvc.yaml", encoding="utf-8")) or {}
except FileNotFoundError:
    sys.exit("dvc.yaml нет — стадия tokenize не объявлена, датасет вне версионирования")
st = (d.get("stages") or {}).get("tokenize")
if not st:
    sys.exit("в dvc.yaml нет стадии tokenize")
norm = lambda x: str(x).rstrip("/")
deps = {norm(x) for x in (st.get("deps") or [])}
outs = {norm(x) for x in (st.get("outs") or [])}
need = [p["data"]["train_jsonl"], p["data"]["val_jsonl"], "src/tokenize_data.py", "src/prompt.py"]
missing = [x for x in need if norm(x) not in deps]
if missing:
    sys.exit("в deps не хватает: " + ", ".join(missing)
             + " — при их изменении dvc не пересчитает стадию")
if norm(p["data"]["out_dir"]) not in outs:
    sys.exit(f"outs не содержит {p['data']['out_dir']} — токенизированный датасет не версионируется")
mets = {norm(list(m)[0] if isinstance(m, dict) else m) for m in (st.get("metrics") or [])}
if "metrics/tokenize.json" not in mets:
    sys.exit("metrics/tokenize.json не объявлен в metrics стадии — dvc metrics diff его не увидит")
print(f"deps {len(deps)}, outs {sorted(outs)}, metrics/tokenize.json объявлен")
PY
); then
  ok "стадия описана: $out"
else
  fail "dvc.yaml не описывает стадию tokenize либо расходится с params.yaml"
  why "$out"
fi

echo
echo "8. Разбор дефектов написан"
DEFECTS=docs/defects.md
if [ ! -s "$DEFECTS" ]; then
  fail "нет $DEFECTS — разбор найденных дефектов не написан (или закоммичен не тот файл)"
else
  heads=$(grep -cE '^#{1,3} ' "$DEFECTS" || true)
  words=$(wc -w < "$DEFECTS" | tr -d ' ')
  if [ "$heads" -lt 4 ] || [ "$words" -lt 200 ]; then
    fail "$DEFECTS слишком короткий: разделов $heads, слов $words — по разделу на каждый дефект, с числами"
  else
    ok "$DEFECTS: разделов $heads, слов $words"
  fi
fi

echo
echo "9. Гигиена репозитория"
tracked_data=$(git ls-files data/ 2>/dev/null | grep -v '\.gitignore$' || true)
junk=$(git ls-files 2>/dev/null | grep -E '(^|/)(\.DS_Store|__pycache__|\.venv|.*\.bak|.*\.orig|~\$.*|\.check_.*\.log)$' || true)
big=$(git ls-files -z 2>/dev/null | xargs -0 -I{} sh -c 'test -f "{}" && s=$(wc -c < "{}") && [ "$s" -gt 5242880 ] && echo "{} ($((s/1048576)) МБ)"' 2>/dev/null || true)
if [ -z "$tracked_data" ] && [ -z "$junk" ] && [ -z "$big" ]; then
  ok "в git нет данных, мусора и файлов тяжелее 5 МБ"
else
  fail "в git попало лишнее"
  [ -n "$tracked_data" ] && echo "$tracked_data" | sed 's/^/      данные: /'
  [ -n "$junk" ] && echo "$junk" | sed 's/^/      мусор: /'
  [ -n "$big" ] && echo "$big" | sed 's/^/      тяжёлый файл: /'
fi

echo
if command -v shasum >/dev/null 2>&1; then fp=$(shasum -a 256 "$SELF" | cut -c1-12); else fp=$(sha256sum "$SELF" | cut -c1-12); fi
printf 'отпечаток tests/check.sh: \033[35m%s\033[0m — на видео сдачи должен совпадать с выданным\n' "$fp"

echo
if [ "$fails" -eq 0 ]; then
  printf '\033[32mВсе проверки пройдены.\033[0m Отчёт: docs/tokenize_report.md, метрики: metrics/tokenize.json\n\n'
else
  printf '\033[31mПровалено проверок: %s\033[0m\n\n' "$fails"
  exit 1
fi
