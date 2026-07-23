"""API endpoint tests for the FastAPI app.

These exercise the request/response contracts of /api/query, /api/session/{id},
/api/courses, and / against the isolated `test_app`/`client` fixtures in
conftest.py (a mocked RAGSystem, no real static assets), rather than importing
app.py directly.
"""


class TestQueryEndpoint:
    def test_creates_session_when_none_provided(self, client, mock_rag_system):
        response = client.post("/api/query", json={"query": "What is MCP?"})

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-id"
        mock_rag_system.query.assert_called_once_with("What is MCP?", "test-session-id")

    def test_uses_provided_session_id(self, client, mock_rag_system):
        response = client.post(
            "/api/query",
            json={"query": "What is MCP?", "session_id": "existing-session"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "existing-session"
        mock_rag_system.query.assert_called_once_with("What is MCP?", "existing-session")
        mock_rag_system.session_manager.create_session.assert_not_called()

    def test_returns_answer_and_sources(self, client):
        response = client.post("/api/query", json={"query": "What is MCP?"})

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "This is a test answer."
        assert data["sources"] == [
            {"text": "Course A - Lesson 1", "link": "https://example.com/lesson1"}
        ]

    def test_source_without_link_defaults_to_none(self, client, mock_rag_system):
        mock_rag_system.query.return_value = (
            "An answer with no link.",
            [{"text": "Course A - Lesson 2"}],
        )

        response = client.post("/api/query", json={"query": "What is MCP?"})

        assert response.status_code == 200
        assert response.json()["sources"] == [{"text": "Course A - Lesson 2", "link": None}]

    def test_missing_query_field_returns_422(self, client):
        response = client.post("/api/query", json={})

        assert response.status_code == 422

    def test_rag_system_error_returns_500(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("boom")

        response = client.post("/api/query", json={"query": "What is MCP?"})

        assert response.status_code == 500
        assert response.json()["detail"] == "boom"


class TestClearSessionEndpoint:
    def test_clears_session_successfully(self, client, mock_rag_system):
        response = client.delete("/api/session/some-session-id")

        assert response.status_code == 200
        assert response.json() == {"success": True}
        mock_rag_system.session_manager.clear_session.assert_called_once_with("some-session-id")

    def test_error_returns_500(self, client, mock_rag_system):
        mock_rag_system.session_manager.clear_session.side_effect = RuntimeError("session gone")

        response = client.delete("/api/session/some-session-id")

        assert response.status_code == 500
        assert response.json()["detail"] == "session gone"


class TestCourseStatsEndpoint:
    def test_returns_analytics(self, client):
        response = client.get("/api/courses")

        assert response.status_code == 200
        assert response.json() == {
            "total_courses": 2,
            "course_titles": ["Course A", "Course B"],
        }

    def test_error_returns_500(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("db down")

        response = client.get("/api/courses")

        assert response.status_code == 500
        assert response.json()["detail"] == "db down"


class TestStaticFrontend:
    def test_root_serves_index_html(self, client):
        response = client.get("/")

        assert response.status_code == 200
        assert "Test Frontend" in response.text
