.PHONY: all install run debug lint clean fclean re

all: install run

install:
	@echo "==> Creating virtual environment and Installing dependencies useing uv sync"
	@uv sync

run:
	@echo "==> Running"
	@uv run python -m src

debug:
	@echo "==> Starting in debug mode..."
	@uv run python -m pdb -m src

lint:
	@echo "==> Running flake8..."
	@flake8 . --extend-exclude=data,tests,moulinette,.venv || true
	@echo "==> Running mypy..."
	@mypy .\
		--warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs\
		--exclude '(^|/)(data|tests|moulinette)/' \
		|| true
		
	@echo "==> Lint complete."

clean:
	@echo "==> Cleaning caches..."
	@find src -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find src -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find src -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "==> Clean complete."

fclean: clean
	@echo "==> Removing virtual environment..."
	@rm -rf .venv
	@echo "==> Full clean complete."

re: fclean all
