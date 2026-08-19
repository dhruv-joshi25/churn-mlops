.PHONY: install install-ui train mlflow api ui test docker up down

install:
	pip install -r requirements.txt

install-ui:
	pip install -r requirements-ui.txt

train:
	python -m src.train --model xgboost --register

mlflow:
	mlflow ui --backend-store-uri ./mlruns --port 5000

api:
	uvicorn src.api.main:app --reload --port 8000

ui:
	streamlit run app/streamlit_app.py --server.port 8501

test:
	pytest -q

docker:
	docker build -t churn-api:local .

up:
	docker compose up --build

down:
	docker compose down -v
