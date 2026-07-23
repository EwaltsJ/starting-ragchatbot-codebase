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
        mock_client.messages.create.return_value = make_response(
            [text_block("Paris is the capital of France.")]
        )

        result = generator.generate_response("What is the capital of France?")

        assert result == "Paris is the capital of France."
        assert mock_client.messages.create.call_count == 1

    def test_tools_and_tool_choice_included_when_tools_provided(
        self, generator_with_mock_client
    ):
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

    def test_conversation_history_embedded_in_system_prompt(
        self, generator_with_mock_client
    ):
        generator, mock_client = generator_with_mock_client
        mock_client.messages.create.return_value = make_response([text_block("answer")])

        generator.generate_response(
            "question", conversation_history="User: hi\nAssistant: hello"
        )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "Previous conversation:" in call_kwargs["system"]
        assert "User: hi" in call_kwargs["system"]

    def test_extract_text_skips_leading_thinking_blocks(
        self, generator_with_mock_client
    ):
        generator, mock_client = generator_with_mock_client
        thinking_block = SimpleNamespace(type="thinking", thinking="reasoning...")
        mock_client.messages.create.return_value = make_response(
            [thinking_block, text_block("final answer")]
        )

        result = generator.generate_response("question")

        assert result == "final answer"

    def test_extract_text_returns_empty_string_when_no_text_block(
        self, generator_with_mock_client
    ):
        generator, mock_client = generator_with_mock_client
        mock_client.messages.create.return_value = make_response(
            [SimpleNamespace(type="thinking", thinking="x")]
        )

        result = generator.generate_response("question")

        assert result == ""


class TestGenerateResponseWithToolUse:
    def test_tool_manager_executes_tool_with_correct_args(
        self, generator_with_mock_client
    ):
        generator, mock_client = generator_with_mock_client
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "[Course A - Lesson 1]\nSome content"

        initial_response = make_response(
            [
                tool_use_block(
                    "search_course_content", {"query": "MCP basics", "lesson_number": 1}
                )
            ],
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

    def test_second_round_includes_tools_and_tool_result_message(
        self, generator_with_mock_client
    ):
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
            "question",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        # Round 2 still offers tools, since Claude may choose to make a second tool call
        second_call_kwargs = mock_client.messages.create.call_args_list[1].kwargs
        assert second_call_kwargs["tools"] == [{"name": "search_course_content"}]
        assert second_call_kwargs["tool_choice"] == {"type": "auto"}

        messages = second_call_kwargs["messages"]
        tool_result_message = messages[-1]
        assert tool_result_message["role"] == "user"
        assert tool_result_message["content"] == [
            {
                "type": "tool_result",
                "tool_use_id": "tool_abc",
                "content": "search result text",
            }
        ]

    def test_no_tool_execution_when_stop_reason_is_not_tool_use(
        self, generator_with_mock_client
    ):
        generator, mock_client = generator_with_mock_client
        tool_manager = MagicMock()
        mock_client.messages.create.return_value = make_response(
            [text_block("direct answer")]
        )

        result = generator.generate_response(
            "question",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        tool_manager.execute_tool.assert_not_called()
        assert result == "direct answer"
        assert mock_client.messages.create.call_count == 1


class TestSequentialToolCalling:
    def test_two_rounds_then_synthesis_call(self, generator_with_mock_client):
        generator, mock_client = generator_with_mock_client
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = ["outline result", "search result"]

        round1_response = make_response(
            [tool_use_block("get_course_outline", {"course_name": "X"}, id_="tool_1")],
            stop_reason="tool_use",
        )
        round2_response = make_response(
            [
                tool_use_block(
                    "search_course_content", {"query": "topic Y"}, id_="tool_2"
                )
            ],
            stop_reason="tool_use",
        )
        synthesis_response = make_response([text_block("final synthesized answer")])
        mock_client.messages.create.side_effect = [
            round1_response,
            round2_response,
            synthesis_response,
        ]

        result = generator.generate_response(
            "compare question",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        assert result == "final synthesized answer"
        assert mock_client.messages.create.call_count == 3
        assert tool_manager.execute_tool.call_count == 2
        tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="X")
        tool_manager.execute_tool.assert_any_call(
            "search_course_content", query="topic Y"
        )

        synthesis_call_kwargs = mock_client.messages.create.call_args_list[2].kwargs
        assert "tools" not in synthesis_call_kwargs
        assert "tool_choice" not in synthesis_call_kwargs

    def test_round_cap_enforced_even_if_synthesis_would_request_tool(
        self, generator_with_mock_client
    ):
        generator, mock_client = generator_with_mock_client
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "result"

        round1_response = make_response(
            [tool_use_block("search_course_content", {"query": "a"}, id_="tool_1")],
            stop_reason="tool_use",
        )
        round2_response = make_response(
            [tool_use_block("search_course_content", {"query": "b"}, id_="tool_2")],
            stop_reason="tool_use",
        )
        # Even though this response looks tool_use-shaped, the synthesis call omits
        # tools entirely, so the API cannot legally return tool_use here in practice;
        # we just confirm the loop never makes a 4th call regardless.
        synthesis_response = make_response(
            [text_block("best effort answer")], stop_reason="tool_use"
        )
        mock_client.messages.create.side_effect = [
            round1_response,
            round2_response,
            synthesis_response,
        ]

        result = generator.generate_response(
            "question",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        assert result == "best effort answer"
        assert mock_client.messages.create.call_count == 3

    def test_tool_execution_error_terminates_loop_and_returns_gracefully(
        self, generator_with_mock_client
    ):
        generator, mock_client = generator_with_mock_client
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = RuntimeError("vector store unavailable")

        round1_response = make_response(
            [tool_use_block("search_course_content", {"query": "a"}, id_="tool_1")],
            stop_reason="tool_use",
        )
        synthesis_response = make_response(
            [text_block("I couldn't complete the search.")]
        )
        mock_client.messages.create.side_effect = [round1_response, synthesis_response]

        result = generator.generate_response(
            "question",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        assert result == "I couldn't complete the search."
        assert mock_client.messages.create.call_count == 2

        synthesis_call_kwargs = mock_client.messages.create.call_args_list[1].kwargs
        tool_result_message = synthesis_call_kwargs["messages"][-1]
        assert tool_result_message["content"][0]["is_error"] is True
        assert (
            "vector store unavailable" in tool_result_message["content"][0]["content"]
        )

    def test_tool_execution_error_falls_back_when_synthesis_has_no_text(
        self, generator_with_mock_client
    ):
        generator, mock_client = generator_with_mock_client
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = RuntimeError("boom")

        round1_response = make_response(
            [tool_use_block("search_course_content", {"query": "a"}, id_="tool_1")],
            stop_reason="tool_use",
        )
        synthesis_response = make_response([])
        mock_client.messages.create.side_effect = [round1_response, synthesis_response]

        result = generator.generate_response(
            "question",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        assert result == AIGenerator.FALLBACK_MESSAGE

    def test_parallel_tool_use_blocks_in_one_round(self, generator_with_mock_client):
        generator, mock_client = generator_with_mock_client
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = ["result A", "result B"]

        round1_response = make_response(
            [
                tool_use_block("search_course_content", {"query": "a"}, id_="tool_1"),
                tool_use_block("search_course_content", {"query": "b"}, id_="tool_2"),
            ],
            stop_reason="tool_use",
        )
        final_response = make_response([text_block("combined answer")])
        mock_client.messages.create.side_effect = [round1_response, final_response]

        result = generator.generate_response(
            "question",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        assert result == "combined answer"
        assert tool_manager.execute_tool.call_count == 2

        second_call_kwargs = mock_client.messages.create.call_args_list[1].kwargs
        tool_result_message = second_call_kwargs["messages"][-1]
        assert tool_result_message["content"] == [
            {"type": "tool_result", "tool_use_id": "tool_1", "content": "result A"},
            {"type": "tool_result", "tool_use_id": "tool_2", "content": "result B"},
        ]

    def test_tool_not_found_string_result_does_not_terminate_loop(
        self, generator_with_mock_client
    ):
        generator, mock_client = generator_with_mock_client
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = [
            "Tool 'bad_tool' not found",
            "second round result",
        ]

        round1_response = make_response(
            [tool_use_block("bad_tool", {"query": "a"}, id_="tool_1")],
            stop_reason="tool_use",
        )
        round2_response = make_response(
            [tool_use_block("search_course_content", {"query": "b"}, id_="tool_2")],
            stop_reason="tool_use",
        )
        synthesis_response = make_response([text_block("final answer")])
        mock_client.messages.create.side_effect = [
            round1_response,
            round2_response,
            synthesis_response,
        ]

        result = generator.generate_response(
            "question",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        assert result == "final answer"
        assert mock_client.messages.create.call_count == 3

        first_result_message = mock_client.messages.create.call_args_list[1].kwargs[
            "messages"
        ][-1]
        assert first_result_message["content"] == [
            {
                "type": "tool_result",
                "tool_use_id": "tool_1",
                "content": "Tool 'bad_tool' not found",
            }
        ]

    def test_system_prompt_describes_two_round_capability(self):
        assert "One tool call per query maximum" not in AIGenerator.SYSTEM_PROMPT
        assert "up to 2 sequential tool calls" in AIGenerator.SYSTEM_PROMPT


@pytest.mark.live
class TestAIGeneratorLive:
    def test_live_call_to_configured_model_succeeds(self, real_config):
        generator = AIGenerator(
            real_config.ANTHROPIC_API_KEY, real_config.ANTHROPIC_MODEL
        )

        result = generator.generate_response("Say hello in exactly one word.")

        assert isinstance(result, str)
        assert result.strip() != ""
