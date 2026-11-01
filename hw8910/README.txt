ДЗ 8+9+10 — проверяемый выпуск и обслуживание модели

Python 3.12. Исходник сохранён коммитом 1837801. tests/test_release.py не менялся.
Полное описание решения и измерений: REPORT.md; семь причин: defects.csv.

БЫСТРАЯ ПРОВЕРКА (без torch, весов, MLflow)
  uv venv --python 3.12 .venv
  uv pip sync --python .venv/bin/python requirements-fast.lock.txt
  make check PYTHON=.venv/bin/python
  make check-fastapi PYTHON=.venv/bin/python
  make check-prefect PYTHON=.venv/bin/python

13 исходных тестов + собственные интеграционные/регрессионные тесты.
HTTP настоящий, выпуск выполняется отдельными subprocess; модель — StubBackend.
Prefect поднимает локальный временный сервер. PREFECT_HOME=.prefect; аналитика
отключена. Для localhost исключён системный прокси. Внешние сервисы не нужны.

КРАСНОЕ ДОКАЗАТЕЛЬСТВО
  .venv/bin/python scripts/reproduce_before.py
Возвращает 1: семь одинаковых сценариев падают по причинам инцидентов на исходном
коммите во временном каталоге. Код рабочего проекта не меняет. Лог: evidence/before.log.
Зелёные логи: evidence/after-stdlib.log, after-fastapi.log, after-prefect.log.

СИНТЕТИЧЕСКИЙ ОПЫТ
  .venv/bin/python lab.py --workspace .lab-demo init
  .venv/bin/python lab.py --workspace .lab-demo run --engine prefect --fail-after-prepare
  .venv/bin/python lab.py --workspace .lab-demo run --engine prefect
  .venv/bin/python lab.py --workspace .lab-demo run --engine prefect --fail-after-pointer
  .venv/bin/python lab.py --workspace .lab-demo run --engine prefect
  .venv/bin/python service.py --workspace .lab-demo --engine fastapi --port 8091 --warmup-seconds 10
Отказы: run --report report-incomplete.json / report-reject.json / report-A.json.
Откат: run --manifest manifest-A.json --report report-A.json, затем перезапуск API.
Старый HTTP-процесс продолжает отвечать загруженной версией до перезапуска.
Для нового независимого опыта нужен новый workspace: init каталог не затирает.

Стандартные CLI-адаптеры lab.py и service.py по умолчанию обходятся стандартной
библиотекой и сохраняют исходные условия запуска. Итоговая архитектура выбирается
--engine prefect и --engine fastapi. У обоих адаптеров единые release/ServiceState,
проверки запросов, backend и HTTP-контракты. create_app принимает backend_factory.
GET /debug доступен в FastAPI только для stub. Реальный backend по умолчанию не
выбирается автоматически, чтобы быстрые тесты никогда не загружали веса.

НАСТОЯЩАЯ МОДЕЛЬ, БЕЗ ПОВТОРНОГО SWEEP
  uv pip sync --python .venv/bin/python requirements.lock.txt
  cd ../hw67
  docker compose -p mlops-hw6 up -d --wait
  cd ../hw8910
  .venv/bin/python real_release.py --workspace .real
  .venv/bin/python lab.py --workspace .real run --engine prefect --manifest manifest-candidate.json --report report-candidate.json
  .venv/bin/python scripts/check_gate.py --workspace .real
  .venv/bin/python service.py --workspace .real --engine fastapi --backend transformers --warmup-seconds 0
  .venv/bin/python scripts/smoke.py --workspace .real

Для отклонённого кандидата run/check_gate возвращают ненулевой код — это ожидаемый
отказ, а не ошибка реализации. real_release сохраняет полную оценку и решение,
даже если кандидат отклонён. Он не меняет MLflow aliases и не запускает обучение.
Новый eval нужен потому, что старые отчёты HW67 не фиксировали весь runtime.
Нельзя вписать новые хеши в старый eval. Новый eval использует ровно тот backend,
который обслуживает API; raw, полные отчёты и компактный release-report сохраняются.
Качество проверяет существующий evaluation.solution.quality_gate из HW67: общие
accuracy/format rate, критические срезы, behavioral-кейсы. Пороги не ослаблены.

При первом запуске явно сохраняется исходная baseline как текущая точка отсчёта.
Её report-baseline.json НЕ объявляется PASS: это удержание существующей базы,
а не новая публикация по candidate gate. Если кандидат отклонён, API загружает
эту базу. Ограничения исходной базы и behavioral тоже видны в полных отчётах.
Повторная оценка не заменяет уже существующий accepted.json отказом кандидата.

ЗАПРОСЫ (после /ready=200)
  curl --noproxy '*' http://127.0.0.1:8091/health
  curl --noproxy '*' http://127.0.0.1:8091/ready
  curl --noproxy '*' -H 'Content-Type: application/json' -d '{"prompt":"Какой компонент Android отображает интерфейс? A: Activity. B: Service.","max_new_tokens":32}' http://127.0.0.1:8091/generate
Реальному API разрешён только оценённый лимит 32; stub допускает целые 1..32.
Вход: непустая строка до 4096 символов, у настоящей модели до 512 токенов вместе
с system/template. Backend последовательный, параллельные запросы ждут lock.
JSON содержит answer, model_id, runtime_sha256, request_id, output_tokens,
generation_seconds, queue_seconds, total_seconds. 503 до загрузки и прогрева.

ЗАМЕР
  .venv/bin/python benchmark.py --url http://127.0.0.1:8091 --output evidence/real-api/benchmark
8 одинаковых по составу запросов при concurrency=1 и ещё 8 при concurrency=2.
Сохраняются все строки, ошибки, таймауты и сводная таблица. Полный JSON-ответ:
измеряется полная задержка; TTFT/ITL/p95 не объявляются.

ТРИ ТРИГГЕРА
Код: make check-prefect PYTHON=.venv/bin/python. Изменён backend/runtime — real_release заново.
Данные: scripts/prepare_data.py --workspace .real; ключ кеша меняется по содержимому/config.
Новый адаптер: real_release.py (полная оценка), затем lab.py run --engine prefect с candidate manifest/report.
Обучение нужно только при изменении train-данных/цели эксперимента; смена golden
сама по себе требует переоценки, а не обучения. Документация не требует обучения.
Локальный CI засчитывается; ci.example.yml — шаблон, удалённый Actions не заявлен.

АРТЕФАКТЫ И ДОСТУП
Веса скачиваются из исходного MLflow training run в .real/model, база — из локального
HF cache по точной revision. При переносе машины сначала восстановить исходные
тома mlops-hw6_postgres/mlops-hw6_artifacts и HF snapshot либо получить доступ к их
копиям; это внешние артефакты, не часть Git. real_release пересоздаёт локальные URI.
В Git только код, маленькие исходные отчёты/raw, хеши и проверяемые журналы.
.venv, .prefect, .lab*, .real*, веса и кеши исключены. Секретов в настройках нет.

Технические основания:
https://docs.prefect.io/v3/concepts/caching
https://reference.prefect.io/prefect/tasks/
https://fastapi.tiangolo.com/advanced/events/
