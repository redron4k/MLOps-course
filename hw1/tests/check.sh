#!/usr/bin/env bash
# Самопроверка домашней работы 1.
# Зелёный check.sh необходим для сдачи, но не достаточен: код читается глазами.
set -uo pipefail
cd "$(dirname "$0")/.."

fails=0
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$1"; fails=$((fails+1)); }

echo
echo "1. Имя модели не захардкожено в коде"
if grep -rqE '"(Qwen|HuggingFaceTB|meta-llama)/' src/ 2>/dev/null; then
  fail "в src/ найдено имя модели строкой — оно должно жить только в params.yaml"
  grep -rnE '"(Qwen|HuggingFaceTB|meta-llama)/' src/ | sed 's/^/      /'
else
  ok "в src/ имён моделей нет"
fi

echo
echo "2. Смена модели в params.yaml не требует правок в коде"
cp params.yaml params.yaml.orig
sed -i.bak 's|^  name: .*|  name: "HuggingFaceTB/SmolLM2-135M-Instruct"|' params.yaml
# Мало убедиться, что не упало: проверяем, что загрузилась именно та модель,
# которая указана в конфиге. Иначе захардкоженное имя проходит незамеченным.
if uv run python -c "
from src.config import load_params
from src.model import load_model
p = load_params()
_, m = load_model(p)
got = str(getattr(m.config, 'name_or_path', ''))
assert p['model']['name'] in got, f'конфиг просит {p[\"model\"][\"name\"]}, а загружено {got}'
" > /dev/null 2>&1; then
  ok "загружается именно та модель, что указана в params.yaml"
else
  fail "загружена не та модель — имя берётся не из конфига"
fi
mv params.yaml.orig params.yaml && rm -f params.yaml.bak

echo
echo "3. Генерация воспроизводима при temperature = 0"
make generate > out1.txt 2>/dev/null
make generate > out2.txt 2>/dev/null
if diff -q out1.txt out2.txt > /dev/null 2>&1; then
  ok "два прогона дали идентичный вывод"
else
  fail "прогоны отличаются — не зафиксирован seed либо включён сэмплинг"
  diff out1.txt out2.txt | head -5 | sed 's/^/      /'
fi
rm -f out1.txt out2.txt

echo
echo "4. Зависимости зафиксированы"
if [ -f uv.lock ]; then
  ok "uv.lock на месте"
else
  fail "нет uv.lock — выполните uv sync и закоммитьте файл"
fi
if grep -qE '^ *"(torch|transformers|accelerate|numpy|pyyaml)" *,' pyproject.toml 2>/dev/null; then
  fail "зависимости без версий в pyproject.toml"
else
  ok "версии зависимостей указаны"
fi

echo
echo "5. Замеры разделены и есть прогрев"
if grep -q "warmup" src/bench.py && grep -q "load_time" src/bench.py; then
  ok "прогрев и отдельный замер загрузки присутствуют"
else
  fail "в bench.py нет прогрева либо загрузка не выделена в отдельный замер"
fi

echo
if [ "$fails" -eq 0 ]; then
  printf '\033[32mВсе проверки пройдены.\033[0m Не забудьте docs/hardware.md с разбором дефектов.\n\n'
else
  printf '\033[31mПровалено проверок: %s\033[0m\n\n' "$fails"
  exit 1
fi
