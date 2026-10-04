.PHONY: install train train-all train-freeze compare plot check clean distclean

install:
	uv sync

# Два прогона: адаптеры на всех слоях и с замороженными первыми 14.
train: train-all train-freeze plot

train-all:
	uv run python -m src.train --variant all_layers

train-freeze:
	uv run python -m src.train --variant freeze14

plot:
	uv run python -m src.plot

# Базовая модель против адаптера на пяти фиксированных промптах.
compare:
	uv run python -m src.compare --variant all_layers

check:
	bash tests/check.sh

# clean — артефакты обучения (адаптеры, метрики, графики); всё пересоздаётся make train.
# distclean — ещё и .venv (~0,8 ГБ): когда ДЗ сдано и место нужнее.
clean:
	rm -rf models metrics/*.json docs/curves.png docs/compare.md .check_*.log
	find . -name __pycache__ -not -path "./.venv/*" -prune -exec rm -rf {} +

distclean: clean
	rm -rf .venv
