# ДЗ 11 — Kubernetes под нагрузкой

Решение продолжает `../hw8910`: тот же интерфейс backend, `StubBackend` и проверка
артефакта `release.verify_artifact`. Новый адаптер жизненного цикла добавляет
потоковый HTTP, вычислительный CPU-имитатор, управляемые кеш/очередь/слоты,
корреляцию запросов и метрики. Все ответы явно содержат `backend_mode=simulated`.
Настоящая модель для максимального балла по условию не требуется.

**Отчёт:** [report.html](report.html). **Критерии и доказательства:** [CHECKLIST.md](CHECKLIST.md). **Ревью:** [REVIEW.md](REVIEW.md).
Фактическая полнота сдачи определяется `results/verification.json` и exit code
`make verify`; незавершённый опыт не считается выполненным по наличию скрипта.

Повтор всей подготовки и измерений: `./scripts/reproduce.sh` (предыдущие данные
сохраняются в `results/archives`, опыты используют настоящее время).

## Быстрый запуск

Нужны работающий Docker Desktop/Engine (рекомендуется 4 CPU, 6–8 GiB RAM),
`kubectl`, `uv`, `curl`, около 6 GiB свободного места для образов и кластера.
Измеренный стенд: Apple Silicon, Docker 14 CPU / 7,75 GiB RAM.
Команды выполняются из `practice/hw11`:

```sh
./scripts/bootstrap.sh                      # kind 0.27.0, Helm 3.17.3 + checksum
uv sync --frozen                            # Python 3.12; точные версии uv.lock
df -h .                                    # зафиксировать ДО создания стенда
docker info --format 'CPU={{.NCPU}} RAM={{.MemTotal}} version={{.ServerVersion}}'
docker system df
.tools/kind create cluster --name hw11 --image kindest/node:v1.32.2 \
  --kubeconfig .tools/kubeconfig --wait 120s
docker build -f Dockerfile -t hw11-api:v2 ..
./scripts/setup.sh
.venv/bin/python scripts/environment.py before
make test lint
.venv/bin/python scripts/run_lab.py           # весь реальный стенд, около 40–45 минут
.venv/bin/python scripts/counterexample.py    # случай ClientIP без перекоса: ещё 3×120 с
.venv/bin/python scripts/calibrate.py         # отдельный одинаковый набор HTTP/Histogram
.venv/bin/python scripts/environment.py after
make report
.venv/bin/python scripts/proofs.py
make verify                                 # ненулевой exit при любом пропуске
```

Внимание к Docker build context: Dockerfile из hw11, context — родитель `practice`,
потому что образ импортирует **исходники предыдущего ДЗ**, а не их копии в hw11.
`../.dockerignore` исключает веса, кеши, `.venv` и логи. Образ содержит только
runtime-зависимости; matplotlib и pytest остаются на хосте.

Все команды kubectl/Helm внутри скриптов явно используют `.tools/kubeconfig` и
namespace `hw11`. Они не используют текущий контекст другого проекта.
Мониторинг ставится в тот же namespace; metrics-server — в `kube-system`
**отдельного kind-кластера hw11**. Флаг `--kubelet-insecure-tls` нужен только
локальным self-signed сертификатам kind и не предназначен для production.

`setup.sh` устанавливает закреплённые Prometheus 3.2.1, Grafana 11.6.0,
kube-state-metrics 2.15.0, metrics-server 0.7.2. Python base image закреплён
manifest digest, `requirements-runtime.lock` фиксирует транзитивные версии.
Фактически использованные imageID всех контейнеров доступны в timeline/inventory.

## Что делает прогон

- **A:** 20 с прогрева с дефектными probes → тот же опыт с исправленными probes;
  два длинных потока перед rollout A→B; B с неисправным прогревом; Helm rollback
  к A; SIGKILL процесса и восстановление; два изолированных диалога после потери Pod.
- **B:** 4 клиентских Pod с разными IP; 2 Service × равномерный/80% трафик ×
  3 повтора по 120 с. 12 запросов/с, одинаковые тексты, max_chunks=4 и расписания.
  Холодный кеш перед каждым запуском, тёплый backend. Исправление — Service без
  ClientIP; его повторные прогоны одновременно служат контрольной группой.
- **C:** CPU-нагрузка, HPA 2→3→2, отдельный повтор горячего ClientIP, перегрузка
  очереди для HighTTFC/HighFailureFraction, ожидание нормализации алертов.
- **Калибровка:** 80 последовательных запросов к одной Kubernetes-реплике,
  снимки Histogram до/после, те же N для клиентского p95 и histogram_quantile.

Параметры: 2 слота/Pod, queueLimit=24, queueTimeout=15 с, LRU 16 префиксов,
hit=0,04 с вычислений, miss=0,25 с вычислений, интервал частей 0,08 с.
В сохранённых измерениях основная B/C использует v1; финальный повтор A — v2.
Backend и стоимости hit/miss одинаковы. V2 исправляет допуск старых сетевых
соединений во время preStop и классификацию HTTP 500 с текстовым телом. Исходники
v1 и хеши сохранены отдельно; чистый повтор из README использует v2 во всех фазах.

Для длинного rollout interval=0,25 с, max_chunks=128, ~32 с/поток, grace=50 с.
На CPU-стенде это части, не токены. TTFC и inter-chunk latency не переименовываются
в TTFT/TPOT. Артефакты `artifacts/model-{A,B}.json` и manifest проходят SHA-256
при каждой загрузке. Изменение ConfigMap приводит к rollout через checksum values.

При ошибке окружения сохраните неполный запуск в отдельный каталог, не смешивайте
два измерения под одним run/request ID. Возобновление частей:

```sh
.venv/bin/python scripts/run_lab.py --phases R  # только rollout/rollback/kill/dialogues
.venv/bin/python scripts/run_lab.py --phases B  # все 12 сравнений
.venv/bin/python scripts/run_lab.py --phases C  # HPA/алерты
```

Полный повтор записывает те же логические имена, поэтому прежде переместите
старые `results` в архив и создайте пустой каталог; храните оригиналы отдельно.
В отчёте нельзя выдавать неудавшийся прогон за успешный. Реальные исходные данные
имеют метки времени; искусственное ускорение длительности 120 с не используется.

## Доступ к API и мониторингу

```sh
kubectl --kubeconfig .tools/kubeconfig -n hw11 port-forward service/grafana 3000:3000
kubectl --kubeconfig .tools/kubeconfig -n hw11 port-forward service/prometheus 9090:9090
kubectl --kubeconfig .tools/kubeconfig -n hw11 port-forward service/api 8000:8000
```

Grafana: `http://localhost:3000`, анонимная роль Viewer, dashboard **HW11 · CPU
simulated inference**. JSON-дэшборд и provisioning сохранены в `monitoring/`.
Port-forward Service выбирает один Pod; **не используйте его для исследования
балансировки**. Опыт B выполняет HTTP непосредственно из клиентских Pod через
ClusterIP; для A используется отдельный bridge Pod с новым соединением на запрос.

```sh
curl -N -H 'Content-Type: application/json' -H 'X-Request-ID: manual-1' \
  -d '{"messages":[{"role":"user","content":"CANARY-MY-DIALOG"}],"session_id":"one","max_chunks":4}' \
  http://localhost:8000/generate
```

`/health` — живость процесса, `/startup` и `/ready` — загрузка и прогрев.
`/metrics` — classic Histogram/Counter/Gauge; `/debug` — метаданные и слоты.
`/drain` и `/reset-cache` — внутренние управляющие endpoints учебного стенда.
В production их надо ограничить отдельно от пользовательского API.

## Измерения и проверки

`results/requests.jsonl` содержит **все попытки**, включая отказы/обрывы/тайм-ауты.
Монотонные часы каждого клиента используются только для его длительностей;
между Pod сравниваются согласованное абсолютное расписание и UTC. До первой
части — TTFC, интервалы — список N−1; для N=1 список пуст. Done-маркер не
считается частью. Успех = HTTP 200 + done + ровно ожидаемое число частей.
Повторы отключены; если их включать, каждому нужен новый attempt/request ID.

`client-logs/`, `schedules/`, `timeline.jsonl`, `pod-logs/`, `commands.jsonl`,
`alerts.jsonl`, `prometheus/` сохраняют исходные доказательства. Нет меток
request/session ID в метриках; они остаются в журналах. P95 вычисляется из
суммарных bucket с сохранённой `le`, а не усредняется между Pod.

`make test` проверяет настоящий локальный HTTP. HTTP-проверки в Kubernetes
находятся непосредственно в `run_lab.py` и используют реальные ответы:
прогрев, версия после отката, восстановление после замены, учёт принудительного
обрыва и изоляция диалогов. `make verify` дополнительно проверяет все исходные
записи и полноту сдачи. Он специально не превращает отсутствие результата
в фиктивный PASS.

## Ресурсы и удаление

На API запрошено 0,4 CPU / 192Mi (2 Pod), лимиты 2 CPU / 384Mi; при HPA 3 Pod —
0,6 CPU / 288Mi, лимиты 3 CPU / 576Mi. Мониторинг ~3×96Mi requests, до 3×384Mi
limits. Четыре клиента + bridge ограничены 128Mi каждый. Prometheus хранит до
256 MB / 2 часов, том ограничен 512Mi; Grafana — 128Mi. Нет PVC с весами.
Docker Desktop имеет общий предел 7,75 GiB. Фактические image sizes, Docker df,
свободное место и размер результатов — в `results/environment-*.json`.
Ориентир собственных загрузок: kind node ~1–1,5 GB, API ~160 MB, мониторинг
~0,5–0,8 GB; Docker может хранить распакованные/сжатые слои одновременно.
Это оценка; итоговый snapshot показывает реальный размер.

```sh
.tools/kind delete cluster --name hw11      # удаляет только наш кластер и его тома
# Необязательно, после проверки отсутствия других использований этих образов:
docker image rm hw11-api:v2 kindest/node:v1.32.2
# Скачанные .tools и .venv удаляются отдельно, если больше не нужны.
```

Не выполняйте `docker system prune --volumes`: на машине есть тома других ДЗ.
В Git добавляются исходники и компактные результаты; `.tools`, `.venv` и
внутреннее состояние контейнеров исключены.

Репозиторий: https://github.com/redron4k/MLOps-course, каталог `hw11`.
Комплект сохраняется отдельным локальным коммитом; его SHA можно получить
командой `git log -1 --format=%H`. Для публикации выполните `git push origin main`
из `practice`. Push и сообщение преподавателю автоматически не делаются.
Результат всех 400 проверок данных: `results/verification.json`; девять HTTP-тестов:
`results/tests.txt`. Для этой сдачи записано 32 353 попытки, включая управляемые
отказы и обрывы; обычные сравниваемые прогоны Service прошли без ошибок.
