.PHONY: install generate bench check clean

install:
	uv sync

generate:
	uv run python -m src.generate

bench:
	uv run python -m src.bench

check:
	bash tests/check.sh

clean:
	rm -rf docs/bench.json out1.txt out2.txt params.yaml.bak
