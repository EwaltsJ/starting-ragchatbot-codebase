import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# The backend modules use flat imports (e.g. `from config import config`) and
# config.py/app.py use paths relative to the backend/ directory, so tests must
# run with backend/ on sys.path and as the working directory.
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(BACKEND_DIR)

from vector_store import VectorStore  # noqa: E402
from config import config as real_app_config  # noqa: E402


@pytest.fixture
def mock_vector_store():
    """A fully mocked VectorStore for isolated CourseSearchTool unit tests."""
    return MagicMock(spec=VectorStore)


@pytest.fixture(scope="session")
def real_config():
    """The real app Config, loaded from .env. Skips live tests if no API key is set."""
    if not real_app_config.ANTHROPIC_API_KEY:
        pytest.skip("ANTHROPIC_API_KEY not set; skipping live test")
    return real_app_config


@pytest.fixture(scope="session")
def real_vector_store():
    """A real VectorStore pointed at the existing backend/chroma_db (read-only queries only)."""
    return VectorStore(
        real_app_config.CHROMA_PATH,
        real_app_config.EMBEDDING_MODEL,
        real_app_config.MAX_RESULTS,
    )
