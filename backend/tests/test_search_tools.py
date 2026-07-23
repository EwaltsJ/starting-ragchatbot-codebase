import pytest

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


def make_results(docs, metas, distances=None):
    return SearchResults(
        documents=docs,
        metadata=metas,
        distances=distances or [0.1] * len(docs),
    )


class TestCourseSearchToolExecute:
    def test_execute_with_lesson_number_formats_header_and_uses_lesson_link(self, mock_vector_store):
        mock_vector_store.search.return_value = make_results(
            ["Some lesson content"],
            [{"course_title": "Intro to MCP", "lesson_number": 2}],
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson2"

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="what is MCP")

        assert result == "[Intro to MCP - Lesson 2]\nSome lesson content"
        mock_vector_store.get_lesson_link.assert_called_once_with("Intro to MCP", 2)
        assert tool.last_sources == [
            {"text": "Intro to MCP - Lesson 2", "link": "https://example.com/lesson2"}
        ]

    def test_execute_without_lesson_number_uses_course_link(self, mock_vector_store):
        mock_vector_store.search.return_value = make_results(
            ["Course-level content"],
            [{"course_title": "Intro to MCP", "lesson_number": None}],
        )
        mock_vector_store.get_course_link.return_value = "https://example.com/course"

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="what is MCP")

        assert result == "[Intro to MCP]\nCourse-level content"
        mock_vector_store.get_course_link.assert_called_once_with("Intro to MCP")
        mock_vector_store.get_lesson_link.assert_not_called()
        assert tool.last_sources == [
            {"text": "Intro to MCP", "link": "https://example.com/course"}
        ]

    def test_execute_passes_filters_through_to_vector_store(self, mock_vector_store):
        mock_vector_store.search.return_value = make_results([], [])

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="what is MCP", course_name="MCP", lesson_number=3)

        mock_vector_store.search.assert_called_once_with(
            query="what is MCP", course_name="MCP", lesson_number=3
        )

    @pytest.mark.parametrize(
        "course_name,lesson_number,expected_suffix",
        [
            (None, None, "No relevant content found."),
            ("MCP", None, "No relevant content found in course 'MCP'."),
            (None, 3, "No relevant content found in lesson 3."),
            ("MCP", 3, "No relevant content found in course 'MCP' in lesson 3."),
        ],
    )
    def test_execute_empty_results_message_variants(
        self, mock_vector_store, course_name, lesson_number, expected_suffix
    ):
        mock_vector_store.search.return_value = make_results([], [])

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="x", course_name=course_name, lesson_number=lesson_number)

        assert result == expected_suffix

    def test_execute_returns_vector_store_error_directly(self, mock_vector_store):
        mock_vector_store.search.return_value = SearchResults.empty("No course found matching 'Nope'")

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="x", course_name="Nope")

        assert result == "No course found matching 'Nope'"

    def test_last_sources_overwritten_across_calls(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)

        mock_vector_store.search.return_value = make_results(
            ["doc1"], [{"course_title": "Course A", "lesson_number": 1}]
        )
        mock_vector_store.get_lesson_link.return_value = "link-a"
        tool.execute(query="x")
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Course A - Lesson 1"

        mock_vector_store.search.return_value = make_results(
            ["doc2"], [{"course_title": "Course B", "lesson_number": 5}]
        )
        mock_vector_store.get_lesson_link.return_value = "link-b"
        tool.execute(query="y")
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Course B - Lesson 5"


class TestToolManagerSourceTracking:
    def test_reset_sources_clears_all_registered_tools(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        manager = ToolManager()
        manager.register_tool(tool)

        mock_vector_store.search.return_value = make_results(
            ["doc"], [{"course_title": "Course A", "lesson_number": 1}]
        )
        mock_vector_store.get_lesson_link.return_value = "link"
        manager.execute_tool("search_course_content", query="x")

        assert manager.get_last_sources() == [{"text": "Course A - Lesson 1", "link": "link"}]

        manager.reset_sources()
        assert manager.get_last_sources() == []


@pytest.mark.live
class TestCourseSearchToolLive:
    def test_execute_against_real_vector_store_does_not_raise(self, real_vector_store):
        tool = CourseSearchTool(real_vector_store)
        result = tool.execute(query="course")

        assert isinstance(result, str)
        assert result != ""
        assert not result.startswith("Search error:")
