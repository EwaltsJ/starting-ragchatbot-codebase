from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ai_generator import AIGenerator


def make_response(content, stop_reason="end_turn"):
    return SimpleNamespace(content=content, stop_reason=stop_reason)


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_use_block(name, input_, id_="tool_1"):
    return SimpleNamespace(type="tool_use", name=name, input=input_, id=id_)


@pytest.fixture
def generator_with_mock_client():
    with patch("ai_generator.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        generator = AIGenerator(api_key="test-key", model="test-model")
        yield generator, mock_client


class TestGenerateResponseWithoutToolUse:
    def test_direct_response_makes_single_api_call(self, generator_with_mock_client):
        generator, mock_client = generator_with_mock_client
        mock_client.messages.create.return_value = make_response([text_block("Paris is the capital of France.")])

        result = generator.generate_response("What is the capital of France?")

        assert result == "Paris is the capital of France."
        assert mock_client.messages.create.call_count == 1

    def test_tools_and_tool_choice_included_when_tools_provided(self, generator_with_mock_client):
        generator, mock_client = generator_with_mock_client
        mock_client.messages.create.return_value = make_response([text_block("answer")])
        tools = [{"name": "search_course_content"}]

        generator.generate_response("question", tools=tools, tool_manager=MagicMock())

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tool_choice"] == {"type": "auto"}

    def test_no_tools_key_when_tools_not_provided(self, generator_with_mock_client):
        generator, mock_client = generator_with_mock_client
        mock_client.messages.create.return_value = make_response([text_block("answer")])

        generator.generate_response("question")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "tools" not in call_kwargs
        assert "tool_choice" not in call_kwargs

    def test_conversation_history_embedded_in_system_prompt(self, generator_with_mock_client):
        generator, mock_client = generator_with_mock_client
        mock_client.messages.create.return_value = make_response([text_block("answer")])

        generator.generate_response("question", conversation_history="User: hi\nAssistant: hello")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "Previous conversation:" in call_kwargs["system"]
        assert "User: hi" in call_kwargs["system"]

    def test_extract_text_skips_leading_thinking_blocks(self, generator_with_mock_client):
        generator, mock_client = generator_with_mock_client
        thinking_block = SimpleNamespace(type="thinking", thinking="reasoning...")
        mock_client.messages.create.return_value = make_response([thinking_block, text_block("final answer")])

        result = generator.generate_response("question")

        assert result == "final answer"

    def test_extract_text_returns_empty_string_when_no_text_block(self, generator_with_mock_client):
        generator, mock_client = generator_with_mock_client
        mock_client.messages.create.return_value = make_response([SimpleNamespace(type="thinking", thinking="x")])

        result = generator.generate_response("question")

        assert result == ""


class TestGenerateResponseWithToolUse:
    def test_tool_manager_executes_tool_with_correct_args(self, generator_with_mock_client):
        generator, mock_client = generator_with_mock_client
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "[Course A - Lesson 1]\nSome content"

        initial_response = make_response(
            [tool_use_block("search_course_content", {"query": "MCP basics", "lesson_number": 1})],
            stop_reason="tool_use",
        )
        final_response = make_response([text_block("Synthesized answer")])
        mock_client.messages.create.side_effect = [initial_response, final_response]

        result = generator.generate_response(
            "What is covered in lesson 1?",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="MCP basics", lesson_number=1
        )
        assert result == "Synthesized answer"
        assert mock_client.messages.create.call_count == 2

    def test_followup_call_omits_tools_and_includes_tool_result_message(self, generator_with_mock_client):
        generator, mock_client = generator_with_mock_client
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "search result text"

        initial_response = make_response(
            [tool_use_block("search_course_content", {"query": "x"}, id_="tool_abc")],
            stop_reason="tool_use",
        )
        final_response = make_response([text_block("answer")])
        mock_client.messages.create.side_effect = [initial_response, final_response]

        generator.generate_response(
            "question", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
        )

        final_call_kwargs = mock_client.messages.create.call_args_list[1].kwargs
        assert "tools" not in final_call_kwargs
        assert "tool_choice" not in final_call_kwargs

        messages = final_call_kwargs["messages"]
        tool_result_message = messages[-1]
        assert tool_result_message["role"] == "user"
        assert tool_result_message["content"] == [
            {"type": "tool_result", "tool_use_id": "tool_abc", "content": "search result text"}
        ]

    def test_no_tool_execution_when_stop_reason_is_not_tool_use(self, generator_with_mock_client):
        generator, mock_client = generator_with_mock_client
        tool_manager = MagicMock()
        mock_client.messages.create.return_value = make_response([text_block("direct answer")])

        result = generator.generate_response(
            "question", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
        )

        tool_manager.execute_tool.assert_not_called()
        assert result == "direct answer"
        assert mock_client.messages.create.call_count == 1


@pytest.mark.live
class TestAIGeneratorLive:
    def test_live_call_to_configured_model_succeeds(self, real_config):
        generator = AIGenerator(real_config.ANTHROPIC_API_KEY, real_config.ANTHROPIC_MODEL)

        result = generator.generate_response("Say hello in exactly one word.")

        assert isinstance(result, str)
        assert result.strip() != ""
