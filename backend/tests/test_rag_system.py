from dataclasses import replace

import pytest

from config import config as real_app_config
from rag_system import RAGSystem


@pytest.fixture
def rag_system_real_retrieval():
    """A real RAGSystem wired to the actual chroma_db, but with a fake API key.

    AIGenerator.__init__ never makes a network call, so the fake key is safe here.
    generate_response is monkeypatched per-test so no network call ever happens.
    """
    test_config = replace(real_app_config, ANTHROPIC_API_KEY="test-key")
    return RAGSystem(test_config)


class TestRAGSystemQuery:
    def test_query_with_tool_use_returns_sources_and_resets_them(
        self, rag_system_real_retrieval, monkeypatch
    ):
        rag = rag_system_real_retrieval

        def fake_generate_response(query, conversation_history=None, tools=None, tool_manager=None):
            tool_manager.execute_tool("search_course_content", query="course content")
            return "Here is the synthesized answer."

        monkeypatch.setattr(rag.ai_generator, "generate_response", fake_generate_response)

        answer, sources = rag.query("What does this course cover?")

        assert answer == "Here is the synthesized answer."
        assert isinstance(sources, list)
        for source in sources:
            assert "text" in source
        # Sources must be reset after being read, so a second query with no tool
        # call doesn't leak sources from the first call.
        assert rag.tool_manager.get_last_sources() == []

    def test_query_without_tool_use_returns_no_sources(self, rag_system_real_retrieval, monkeypatch):
        rag = rag_system_real_retrieval

        def fake_generate_response(query, conversation_history=None, tools=None, tool_manager=None):
            return "General knowledge answer, no search needed."

        monkeypatch.setattr(rag.ai_generator, "generate_response", fake_generate_response)

        answer, sources = rag.query("What is 2+2?")

        assert answer == "General knowledge answer, no search needed."
        assert sources == []

    def test_query_updates_session_history(self, rag_system_real_retrieval, monkeypatch):
        rag = rag_system_real_retrieval
        monkeypatch.setattr(
            rag.ai_generator, "generate_response", lambda **kwargs: "an answer"
        )
        session_id = rag.session_manager.create_session()

        rag.query("first question", session_id=session_id)

        history = rag.session_manager.get_conversation_history(session_id)
        assert "first question" in history
        assert "an answer" in history

    def test_query_passes_tool_definitions_to_ai_generator(
        self, rag_system_real_retrieval, monkeypatch
    ):
        rag = rag_system_real_retrieval
        captured = {}

        def fake_generate_response(query, conversation_history=None, tools=None, tool_manager=None):
            captured["tools"] = tools
            captured["tool_manager"] = tool_manager
            return "answer"

        monkeypatch.setattr(rag.ai_generator, "generate_response", fake_generate_response)
        rag.query("a content question")

        tool_names = {t["name"] for t in captured["tools"]}
        assert tool_names == {"search_course_content", "get_course_outline"}
        assert captured["tool_manager"] is rag.tool_manager


@pytest.mark.live
class TestRAGSystemLive:
    def test_live_content_query_does_not_raise_and_returns_answer(self, real_config):
        rag = RAGSystem(real_config)

        answer, sources = rag.query("What topics are covered in this course?")

        assert isinstance(answer, str)
        assert answer.strip() != ""
        assert isinstance(sources, list)


@pytest.mark.live
class TestQueryEndpointLive:
    def test_live_query_endpoint_returns_200_for_content_question(self, real_config):
        from starlette.testclient import TestClient
        from app import app

        client = TestClient(app)
        response = client.post(
            "/api/query", json={"query": "What topics are covered in this course?"}
        )

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["answer"].strip() != ""
        assert isinstance(data["sources"], list)
        assert data["session_id"]
