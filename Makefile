PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
CONFIG ?= configs/config.yaml

.PHONY: setup data eda train evaluate report test lint format run docker-build all

setup:
	python3 -m venv .venv
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -e ".[dev]"

data:
	$(PY) -m credit_risk.data --config $(CONFIG)

eda: data
	$(PY) -m credit_risk.eda --config $(CONFIG)

# Stages are resumable: finished stages are skipped unless FORCE=1.
FORCE_FLAG = $(if $(FORCE),--force,)

train: data
	$(PY) -m credit_risk.pipeline --config $(CONFIG) --stage train $(FORCE_FLAG)

evaluate:
	$(PY) -m credit_risk.pipeline --config $(CONFIG) --stage evaluate $(FORCE_FLAG)

report:
	$(PY) scripts/make_data_card.py
	$(PY) scripts/make_report.py --config $(CONFIG)
	$(PY) scripts/make_readme.py

test:
	$(PY) -m pytest

lint:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

format:
	.venv/bin/ruff format .
	.venv/bin/ruff check --fix .

run:
	$(PY) -m uvicorn credit_risk.api.main:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t credit-risk:latest .

all: setup
	$(MAKE) data eda
	$(MAKE) train FORCE=1
	$(MAKE) evaluate FORCE=1
	$(MAKE) report test lint
