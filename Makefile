.PHONY: install run ui test import train

install:
	python -m pip install -r requirements.txt

run:
	uvicorn app.main:app --reload

ui:
	streamlit run streamlit_app.py

test:
	pytest -q

import:
	python scripts/import_dataset.py data/dataset.csv

train:
	python scripts/train_linear_heads.py
