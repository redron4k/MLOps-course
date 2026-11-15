"""Produce a compact rubric map from the strict delivery gate."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1];r=root/'results'
verify=json.loads((r/'verification.json').read_text())
checks=verify['checks']
failed=[c['name'] for c in checks if not c['passed']]
rows=[
('Стенд и Helm',2,'chart/, Dockerfile, requirements-runtime.lock, artifacts/, results/environment-*.json; восстановление: checks/replacement-dialogues.json'),
('Прогрев и выпуск',2,'checks/A-warmup-*.json; checks/unready-B.json; checks/rollback-version.json; A-rollout и A-rollout-long-* в requests.jsonl; timeline.jsonl; figures/1.svg'),
('Балансировка',2,'12 B-*-{1,2,3} прогонов по 120 с; schedules/; clients.json; network*.json; trace-proofs.json; дополнительный контрпример 3×120 с; figures/2.svg и 3.svg'),
('Метрики и HPA',2,'monitoring/; checks/metrics-api.json; checks/hpa-*.json; timeline.jsonl; alerts.jsonl; prometheus/; histogram-comparison.json; figures/4.svg'),
('Проверки и защита',2,'tests/test_service.py; results/tests.txt; результаты реальных HTTP-проверок run_lab.py; verification.json; 4 SVG; report.html; REVIEW.md с первичным источником Google / llm-d'),
]
text='# Проверка состава сдачи ДЗ 11\n\n'
text+=f'Строгий gate: **{verify["passed"]} успешно, {verify["failed"]} неуспешно**. Баллы ниже — вес критерия в задании, а не обещание оценки.\n\n'
text+='| Критерий | Вес | Доказательства |\n|---|---:|---|\n'
for name,weight,evidence in rows:text+=f'| {name} | {weight} | {evidence} |\n'
text+='\nСам материал лекции 11 отсутствует; совпадение выбранного первичного кейса с лекцией проверить невозможно. Остальные ограничения и точные результаты приведены в report.html.\n'
if failed:text+='\nНезавершённые проверки:\n\n'+''.join('- '+c+'\n' for c in failed)
(root/'CHECKLIST.md').write_text(text)
print('Wrote CHECKLIST.md')
