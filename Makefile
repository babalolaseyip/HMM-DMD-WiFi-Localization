.PHONY: install test clean lint format

install:
	pip install -r requirements/base.txt
	pip install -e .

install-dev:
	pip install -r requirements/dev.txt
	pip install -e .

test:
	pytest -v --cov=. --cov-report=html

lint:
	flake8 models/ src/ tests/

format:
	black models/ src/ tests/ scripts/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .coverage htmlcov/

docker-build:
	docker build -t hmm-dmd-localization .

docker-run:
	docker run -v $(pwd)/results:/app/results hmm-dmd-localization
