PYTHON ?= python3
.PHONY: check
check:
	$(PYTHON) -m compileall -q release.py tests
	$(PYTHON) -m unittest discover -s tests -v
