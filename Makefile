.PHONY: format-code

format-code:
	poetry run black python/
	poetry run flake8 python/
	poetry run mypy python/

.PHONY: tests
tests:
	poetry run pytest python/tests/