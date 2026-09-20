#!/usr/bin/env bash
# Самопроверка домашней работы 3.
# Зелёный check.sh необходим для сдачи, но не достаточен: код читается глазами.
set -uo pipefail
cd "$(dirname "$0")/.."

fails=0
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$1"; fails=$((fails+1)); }

RUN="uv run"
SANDBOX="data/_check"
restore_params() { [ -f params.yaml.orig ] && mv params.yaml.orig params.yaml; rm -f params.yaml.bak; }
trap 'restore_params; rm -rf "$SANDBOX"' EXIT

echo
echo "1. Данные не в git и под управлением DVC"
tracked=$(git ls-files data/ 2>/dev/null | wc -l | tr -d ' ')
if [ "$tracked" = "0" ]; then
  ok "git не отслеживает ни одного файла в data/"
else
  fail "в git закоммичено файлов данных: $tracked"
  git ls-files data/ | head -5 | sed 's/^/      /'
fi
if grep -qE '^data/?$' .gitignore 2>/dev/null; then
  ok "data/ перечислен в .gitignore проекта"
else
  fail "в .gitignore нет data/ — данные утекут в git при первом же git add ."
fi
if [ -n "$($RUN dvc remote list 2>/dev/null)" ]; then
  ok "remote DVC настроен: $($RUN dvc remote list | head -1 | tr -s ' ')"
else
  fail "remote DVC не настроен — выгружать данные некуда, они останутся только в git"
fi
missing=$($RUN python - <<'PY'
import yaml
params = yaml.safe_load(open("params.yaml", encoding="utf-8"))
pipeline = yaml.safe_load(open("dvc.yaml", encoding="utf-8"))
outs = set()
for stage in pipeline["stages"].values():
    for item in stage.get("outs", []):
        outs.add(item if isinstance(item, str) else next(iter(item)))
want = [params["paths"][k] for k in ("raw", "clean", "train", "val", "test")]
print(" ".join(p for p in want if p not in outs))
PY
)
if [ -z "$missing" ]; then
  ok "все артефакты из params.paths объявлены в outs dvc.yaml"
else
  fail "не объявлены в outs dvc.yaml (значит, не версионируются): $missing"
fi

echo
echo "2. Дедупликация ловит не только буквальные повторы"
cp params.yaml params.yaml.orig
mkdir -p "$SANDBOX"
sed -i.bak \
  -e "s|^  raw: .*|  raw: \"$SANDBOX/raw.jsonl\"|" \
  -e "s|^  clean: .*|  clean: \"$SANDBOX/clean.jsonl\"|" \
  -e "s|^  metrics_clean: .*|  metrics_clean: \"$SANDBOX/clean.json\"|" \
  params.yaml
$RUN python - "$SANDBOX/raw.jsonl" <<'PY'
import json, sys
SYS = "Ты отвечаешь на русскоязычные тестовые вопросы НМО."
def ex(i, topic, user, answer="Ответ: 1"):
    return {"id": f"chk_{i}", "topic": topic,
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": user},
                         {"role": "assistant", "content": answer}]}
base = ("Тема:\nАртериальная гипертензия у взрослых\n\nВопрос:\nКакой класс препаратов "
        "рекомендован как терапия первой линии пациенту с артериальной гипертензией "
        "второй степени и сахарным диабетом второго типа?\n\nВарианты ответа:\n"
        "0. тиазидный диуретик\n1. ингибитор ангиотензинпревращающего фермента\n"
        "2. дигидропиридиновый блокатор кальциевых каналов")
# Тот же вопрос, набранный иначе: другие пробелы и нумерация вариантов.
# Точная дедупликация его не видит — тексты различаются, — а near-dup обязан.
near = (base.replace("Вопрос:\n", "Вопрос:\n  ")
            .replace("второго типа?", "второго типа :")
            .replace("0. тиазидный", "0) тиазидный")
            .replace("1. ингибитор", "1) ингибитор")
            .replace("2. дигидропиридиновый", "2) дигидропиридиновый"))
pii = ("Тема:\nДиспансеризация\n\nВопрос:\nПациент, дата рождения 12.03.1975, "
       "телефон +7 (999) 123-45-67, почта ivanov@example.ru. Какой объём "
       "обследования показан?\n\nВарианты ответа:\n0. краткий\n1. расширенный")
rows = [ex(1, "Артериальная гипертензия", base),
        ex(2, "Артериальная гипертензия", base),        # точный дубль
        ex(3, "Артериальная гипертензия", near),        # почти-дубль
        ex(4, "Диспансеризация", pii),                  # телефон, почта, дата рождения
        ex(5, "Пневмония", "Тема:\nПневмония\n\nВопрос:\nВозбудитель внебольничной "
                           "пневмонии чаще всего?\n\nВарианты ответа:\n0. пневмококк\n1. клебсиелла")]
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
PY
if $RUN python -m src.clean > "$SANDBOX/log" 2>&1; then
  read -r n_exact n_near n_pii <<<"$($RUN python -c "
import json; m = json.load(open('$SANDBOX/clean.json', encoding='utf-8'))
print(m['dropped_exact_dup'], m['dropped_near_dup'], sum(m['pii_hits'].values()))")"
  if [ "${n_exact:-0}" -ge 1 ]; then
    ok "точный дубль удалён (dropped_exact_dup = $n_exact)"
  else
    fail "точный дубль пережил очистку"
  fi
  if [ "${n_near:-0}" -ge 1 ]; then
    ok "почти-дубль удалён (dropped_near_dup = $n_near)"
  else
    fail "почти-дубль пережил очистку — дедупликация только точная"
  fi
  if [ "${n_pii:-0}" -ge 3 ]; then
    ok "телефон, почта и дата рождения замаскированы (совпадений $n_pii)"
  else
    fail "чистка ПДн не сработала: подложено 3 совпадения, найдено ${n_pii:-0}"
  fi
else
  fail "стадия clean упала на корректном входе"
  tail -5 "$SANDBOX/log" | sed 's/^/      /'
fi

echo
echo "3. Битый JSONL валит стадию clean с внятной ошибкой"
$RUN python - "$SANDBOX/raw.jsonl" <<'PY'
import json, sys
SYS = "Ты отвечаешь на русскоязычные тестовые вопросы НМО."
good = {"id": "chk_ok", "topic": "Пневмония",
        "messages": [{"role": "system", "content": SYS},
                     {"role": "user", "content": "Тема:\nПневмония\n\nВопрос:\nВозбудитель "
                      "внебольничной пневмонии чаще всего?\n\nВарианты ответа:\n0. пневмококк"},
                     {"role": "assistant", "content": "Ответ: 0. пневмококк"}]}
bad = json.loads(json.dumps(good))
bad["id"] = "chk_bad"
bad["messages"][2]["role"] = "user"      # ролей assistant нет — пример невалиден
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    for r in (good, good, bad):
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
PY
if $RUN python -m src.clean > "$SANDBOX/log" 2>&1; then
  fail "стадия clean прошла на битом JSONL — валидация схемы не гейт, а украшение"
else
  if grep -qE 'raw\.jsonl:3' "$SANDBOX/log"; then
    ok "стадия упала и назвала номер битой строки"
    grep -oE '[^ ]*raw\.jsonl:3:.*' "$SANDBOX/log" | head -1 | cut -c1-120 | sed 's/^/      /'
  else
    fail "стадия упала, но в сообщении нет номера строки — искать битый пример нечем"
    tail -3 "$SANDBOX/log" | sed 's/^/      /'
  fi
fi
restore_params
rm -rf "$SANDBOX"

echo
echo "4. Смена версии данных реально пересчитывает пайплайн"
was=$($RUN python -c "import yaml; print(yaml.safe_load(open('params.yaml', encoding='utf-8'))['collect']['version'])")
$RUN python scripts/set_version.py v1 > /dev/null
$RUN dvc repro > /dev/null 2>&1
h1=$($RUN python -c "import hashlib,sys; print(hashlib.md5(open('data/clean.jsonl','rb').read()).hexdigest())" 2>/dev/null)
r1=$($RUN python -c "import json; print(json.load(open('metrics/clean.json',encoding='utf-8'))['rows_in'])" 2>/dev/null)
$RUN python scripts/set_version.py v2 > /dev/null
$RUN dvc repro > "$SANDBOX.log" 2>&1
h2=$($RUN python -c "import hashlib,sys; print(hashlib.md5(open('data/clean.jsonl','rb').read()).hexdigest())" 2>/dev/null)
r2=$($RUN python -c "import json; print(json.load(open('metrics/clean.json',encoding='utf-8'))['rows_in'])" 2>/dev/null)
if [ -n "$h1" ] && [ "$h1" != "$h2" ]; then
  ok "clean.jsonl пересчитан: v1 ${h1:0:8} ($r1 строк) → v2 ${h2:0:8} ($r2 строк)"
else
  fail "выход стадии не изменился при смене версии — deps/params в dvc.yaml заданы неверно"
  echo "      v1 $h1 ($r1 строк) / v2 $h2 ($r2 строк)"
fi
$RUN python scripts/set_version.py "$was" > /dev/null   # версия восстановлена
$RUN dvc repro > /dev/null 2>&1
rm -f "$SANDBOX.log"

echo
echo "5. Контаминации между train и test нет"
if $RUN python scripts/check_contamination.py > "$SANDBOX.log" 2>&1; then
  ok "$(grep -E 'near-dup|пересечение по id' "$SANDBOX.log" | tr -s ' ' | paste -sd'; ' -)"
else
  fail "проверка контаминации нашла пересечения train/test"
  grep -E 'пересечение|near-dup|КОНТАМ' "$SANDBOX.log" | sed 's/^/      /'
fi
rm -f "$SANDBOX.log"

echo
echo "6. Гейт разнообразия падает на вырожденном наборе"
# Подсовываем набор, который проходит схему, дедупликацию и сплит: строки разные,
# дублей нет, битого JSON нет. Вырожден он как НАБОР — один системный промпт,
# одна тема, ответы одной длины. Ни одна из предыдущих стадий этого не видит.
mkdir -p "$SANDBOX"
$RUN python - "$SANDBOX/degenerate.jsonl" <<'PY'
import json, sys
SYS = "Ты отвечаешь на вопросы. Отвечай одним словом."
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    for i in range(1200):
        fh.write(json.dumps({
            "id": f"deg_{i:05d}",
            "topic": "Одна-единственная тема - 2025",
            "messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": f"Вопрос номер {i} про один и тот же предмет?"},
                {"role": "assistant", "content": f"Ответ: {i % 4}. вариант"},
            ],
        }, ensure_ascii=False) + "\n")
PY
cp params.yaml params.yaml.orig
$RUN python - <<'PY'
import re, pathlib
p = pathlib.Path("params.yaml")
s = p.read_text(encoding="utf-8")
s = re.sub(r'^(\s*clean:\s*")data/clean\.jsonl(")', r'\1data/_check/degenerate.jsonl\2', s, flags=re.M)
p.write_text(s, encoding="utf-8")
PY
if $RUN python -m src.diversity > "$SANDBOX.log" 2>&1; then
  fail "стадия diversity приняла вырожденный набор — гейт не работает"
  sed -n '1,4p' "$SANDBOX.log" | sed 's/^/      /'
else
  hits=$(grep -cE 'системных промптов|групп |длин ответа|одну и ту же длину' "$SANDBOX.log" || true)
  if [ "$hits" -ge 2 ]; then
    ok "гейт упал и назвал $hits нарушенных порога — разбираться есть по чему"
  else
    fail "гейт упал, но не объяснил, что именно не так"
    sed -n '1,6p' "$SANDBOX.log" | sed 's/^/      /'
  fi
fi
restore_params
rm -f "$SANDBOX.log"

echo
echo "7. Гейт разнообразия пропускает нормальный датасет"
if $RUN python -m src.diversity > "$SANDBOX.log" 2>&1; then
  ok "$(tail -1 "$SANDBOX.log")"
else
  fail "гейт завернул ваш собственный датасет — пороги или данные требуют разбора"
  sed -n '1,6p' "$SANDBOX.log" | sed 's/^/      /'
fi
rm -f "$SANDBOX.log"

echo
echo "8. Гигиена репозитория"
junk=$(git ls-files 2>/dev/null | grep -E '(^|/)(\.DS_Store|__pycache__|\.venv|.*\.bak|.*\.orig|~\$.*)$' || true)
big=$(git ls-files -z 2>/dev/null | xargs -0 -I{} sh -c 'test -f "{}" && s=$(wc -c < "{}") && [ "$s" -gt 5242880 ] && echo "{} ($((s/1048576)) МБ)"' 2>/dev/null || true)
if [ -z "$junk" ] && [ -z "$big" ]; then
  ok "в git нет мусора и файлов тяжелее 5 МБ"
else
  fail "в git попало лишнее"
  [ -n "$junk" ] && echo "$junk" | sed 's/^/      мусор: /'
  [ -n "$big" ] && echo "$big" | sed 's/^/      тяжёлый файл: /'
fi

echo
if [ "$fails" -eq 0 ]; then
  printf '\033[32mВсе проверки пройдены.\033[0m Не забудьте docs/datasheet.md и разбор дефектов.\n\n'
else
  printf '\033[31mПровалено проверок: %s\033[0m\n\n' "$fails"
  exit 1
fi
