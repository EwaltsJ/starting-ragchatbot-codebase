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

from config import config as real_app_config  # noqa: E402
from vector_store import VectorStore  # noqa: E402


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


@pytest.fixture
def mock_rag_system():
    """A mocked RAGSystem for API endpoint tests, with sensible default return values."""
    mock = MagicMock()
    mock.session_manager.create_session.return_value = "test-session-id"
    mock.query.return_value = (
        "This is a test answer.",
        [{"text": "Course A - Lesson 1", "link": "https://example.com/lesson1"}],
    )
    mock.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Course A", "Course B"],
    }
    return mock


@pytest.fixture
def test_app(mock_rag_system, tmp_path):
    """A FastAPI app mirroring app.py's routes, without app.py's import-time side effects.

    app.py instantiates a real RAGSystem and mounts ../frontend as static files at
    import time, which is slow and environment-dependent. This fixture defines the
    same endpoint contracts inline against a mocked RAGSystem and a throwaway static
    directory, so endpoint tests stay fast and isolated.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    from typing import List, Optional

    app = FastAPI(title="Course Materials RAG System - Test")

    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class SourceItem(BaseModel):
        text: str
        link: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[SourceItem]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            answer, sources = mock_rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def clear_session(session_id: str):
        try:
            mock_rag_system.session_manager.clear_session(session_id)
            return {"success": True}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    static_dir = tmp_path / "frontend"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html><body>Test Frontend</body></html>")
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


@pytest.fixture
def client(test_app):
    """A TestClient for the isolated test_app, for API endpoint tests."""
    from fastapi.testclient import TestClient

    return TestClient(test_app)
