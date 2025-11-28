.PHONY: install train test clean

install:
	pip install -r requirements.txt

train:
	python train.py

test:
	pytest tests/ -v

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	rm -rf .pytest_cache
