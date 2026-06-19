# Makefile for RAG Document Q&A System

.PHONY: help install run test clean format

help:
	@echo "Available commands:"
	@echo "  make install  - Install the package in development mode"
	@echo "  make run      - Run the Streamlit app"
	@echo "  make test     - Run tests with pytest"
	@echo "  make format   - Format code with black and isort"
	@echo "  make clean    - Remove Python cache files"

install:
	pip install -e .
	pip install -r requirements.txt

run:
	streamlit run app.py

test:
	pytest tests/ -v

format:
	black rag_system_simple.py app.py
	isort rag_system_simple.py app.py

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete