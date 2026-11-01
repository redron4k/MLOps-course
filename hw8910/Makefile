PYTHON ?= python3
.PHONY: check check-fastapi check-prefect
check:
	$(PYTHON) -m compileall -q release.py lab.py service.py api.py pipeline.py backend.py gate.py real_release.py benchmark.py scripts integration_tests tests
	$(PYTHON) scripts/check.py
check-fastapi:
	HTTP_ENGINE=fastapi $(PYTHON) scripts/check.py
check-prefect:
	HTTP_ENGINE=fastapi FLOW_ENGINE=prefect $(PYTHON) scripts/check.py
