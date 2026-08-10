.PHONY: install install-dev test cov lint run

install:
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-dev.txt

test:
	pytest -v

cov:
	pytest --cov=. --cov-report=term-missing

lint:
	ruff check .

run:
	uvicorn api.main:app --reload
