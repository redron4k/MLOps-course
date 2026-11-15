.PHONY: test lint setup run report verify clean

test:
	.venv/bin/python -m pytest -q
lint:
	.tools/helm lint chart
setup:
	./scripts/setup.sh
run:
	.venv/bin/python scripts/run_lab.py
report:
	MPLCONFIGDIR=.tools/matplotlib .venv/bin/python scripts/analyze.py
verify:
	.venv/bin/python scripts/verify_evidence.py
clean:
	.tools/kind delete cluster --name hw11
