#!/bin/bash
# Run all code quality checks: formatting, linting, and tests.
# Does not modify any files - use scripts/format.sh to auto-fix formatting.
set -e

cd "$(dirname "$0")/.."

echo "Checking formatting with black..."
uv run black --check .

echo "Running ruff..."
uv run ruff check .

echo "Running tests..."
uv run pytest backend/tests -m "not live"

echo "All quality checks passed."
