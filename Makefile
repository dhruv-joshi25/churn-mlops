.PHONY: install install-dev install-ui lint format typecheck test coverage \
        train mlflow api ui docker up down

# Runtime install: the package plus the serving stack (model libraries, FastAPI).
install:
	pip install -e ".[serving]"

# Everything above plus pytest, ruff and mypy. What CI runs.
install-dev:
	pip install -e ".[dev]"

# UI only. Deliberately excludes scikit-learn, xgboost, shap and mlflow so the
# portal cannot load a model even by accident — it must go through the API.
install-ui:
	pip install -e ".[ui]"

lint:
	ruff check src tests app

format:
	ruff format src tests app

typecheck:
	mypy

test:
	pytest -q

# Coverage is gated on the platform packages only. The Telco reference
# implementation under churnkit/reference is being deleted module by module as
# the platform replaces it; the gate widens as that happens.
coverage:
	pytest -q --cov=churnkit --cov-report=term-missing
	coverage report --include="src/churnkit/ingest/*" --fail-under=80

train:
	python -m churnkit.reference.train --model xgboost --register

mlflow:
	mlflow ui --backend-store-uri ./mlruns --port 5000

api:
	uvicorn churnkit.reference.api.main:app --reload --port 8000

ui:
	streamlit run app/streamlit_app.py --server.port 8501

docker:
	docker build -t churnkit-api:local .

up:
	docker compose up --build

down:
	docker compose down -v
