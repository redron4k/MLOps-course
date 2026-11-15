# Проверка состава сдачи ДЗ 11

Строгий gate: **400 успешно, 0 неуспешно**. Баллы ниже — вес критерия в задании, а не обещание оценки.

| Критерий | Вес | Доказательства |
|---|---:|---|
| Стенд и Helm | 2 | chart/, Dockerfile, requirements-runtime.lock, artifacts/, results/environment-*.json; восстановление: checks/replacement-dialogues.json |
| Прогрев и выпуск | 2 | checks/A-warmup-*.json; checks/unready-B.json; checks/rollback-version.json; A-rollout и A-rollout-long-* в requests.jsonl; timeline.jsonl; figures/1.svg |
| Балансировка | 2 | 12 B-*-{1,2,3} прогонов по 120 с; schedules/; clients.json; network*.json; trace-proofs.json; дополнительный контрпример 3×120 с; figures/2.svg и 3.svg |
| Метрики и HPA | 2 | monitoring/; checks/metrics-api.json; checks/hpa-*.json; timeline.jsonl; alerts.jsonl; prometheus/; histogram-comparison.json; figures/4.svg |
| Проверки и защита | 2 | tests/test_service.py; results/tests.txt; результаты реальных HTTP-проверок run_lab.py; verification.json; 4 SVG; report.html; REVIEW.md с первичным источником Google / llm-d |

Сам материал лекции 11 отсутствует; совпадение выбранного первичного кейса с лекцией проверить невозможно. Остальные ограничения и точные результаты приведены в report.html.
