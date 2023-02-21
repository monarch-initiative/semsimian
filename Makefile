.PHONY: linters

linters:
	poetry run black python/
	poetry run flake8 python/
	poetry run mypy python/