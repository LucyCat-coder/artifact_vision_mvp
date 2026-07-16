.PHONY: install run run-dev ui test validate dry-run import train

install:
	python -m pip install -r requirements.txt

run:
	python -m uvicorn app.main:app

run-dev:
	python -m uvicorn app.main:app --reload

ui:
	python -m streamlit run streamlit_app.py

test:
	python -m pytest -v

validate:
	python scripts/validate_dataset.py data/dataset/metadata/artifacts.csv

dry-run:
	python scripts/import_dataset.py data/dataset/metadata/artifacts.csv --dry-run

import:
	python scripts/import_dataset.py data/dataset/metadata/artifacts.csv --strict

train:
	python scripts/train_linear_heads.py
