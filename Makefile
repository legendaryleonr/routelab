# Common commands. Create and activate a virtual environment first (see README).
PYTHON ?= python

.PHONY: install data train test

install:
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install --no-deps -e .

data:
	$(PYTHON) -m routelab data download

train:
	$(PYTHON) -m routelab train

test:
	$(PYTHON) -m pytest -q
