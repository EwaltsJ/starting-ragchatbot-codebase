#!/bin/bash
# Lint the codebase with ruff.
set -e

cd "$(dirname "$0")/.."

echo "Running ruff..."
uv run ruff check .
