import anthropic
from typing import List, Optional, Dict, Any, Tuple

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Tool Usage:
- Use `get_course_outline` only when asked to list/enumerate a course's lessons or describe its overall structure (e.g. "outline", "what lessons does this course have", "table of contents")
- Use `search_course_content` for anything about the substance of a lesson — what it teaches, covers, or explains — even if the question mentions a lesson number or name
- **You may make up to 2 sequential tool calls per query when genuinely needed** — for example, call `get_course_outline` to find a lesson's exact title or confirm a course name, then use that result as input to a `search_course_content` call
- Prefer a single tool call whenever it fully answers the question; only make a second call if the first call's result is required to complete the second
- Never make more than 2 tool calls for a single query — after that, answer using whatever information you have gathered
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    MAX_TOOL_ROUNDS = 2
    FALLBACK_MESSAGE = "I wasn't able to complete that request. Please try rephrasing your question."

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [{"role": "user", "content": query}]

        # Fast path: no tools available to loop over
        if not tools or tool_manager is None:
            response = self._call_claude(messages, system_content, tools=tools)
            return self._extract_text(response)

        return self._run_tool_loop(messages, system_content, tools, tool_manager)

    def _run_tool_loop(self, messages: List[Dict[str, Any]], system_content: str,
                        tools: List, tool_manager) -> str:
        """
        Run up to MAX_TOOL_ROUNDS rounds of tool-enabled API calls, executing any
        requested tools and feeding results back to Claude between rounds.

        Terminates when: a round's response has no tool_use blocks, a tool
        execution errors, or MAX_TOOL_ROUNDS rounds have completed. In the
        latter two cases, a final synthesis call (tools omitted) produces the
        answer from whatever context has been gathered so far.
        """
        for round_num in range(1, self.MAX_TOOL_ROUNDS + 1):
            response = self._call_claude(messages, system_content, tools=tools)

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_use_blocks:
                return self._extract_text(response)

            messages = messages + [{"role": "assistant", "content": response.content}]
            tool_result_blocks, any_error = self._execute_tool_use_blocks(tool_use_blocks, tool_manager)
            messages = messages + [{"role": "user", "content": tool_result_blocks}]

            if any_error or round_num == self.MAX_TOOL_ROUNDS:
                break

        final_response = self._call_claude(messages, system_content, tools=None)
        return self._extract_text(final_response) or self.FALLBACK_MESSAGE

    @staticmethod
    def _execute_tool_use_blocks(tool_use_blocks: List, tool_manager) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Execute every tool_use block from one round, returning the resulting
        tool_result blocks (as a single list) and whether any call errored.
        """
        tool_results = []
        any_error = False
        for block in tool_use_blocks:
            try:
                content = tool_manager.execute_tool(block.name, **block.input)
            except Exception as exc:
                any_error = True
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Tool '{block.name}' failed: {exc}",
                    "is_error": True,
                })
                continue
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content,
            })
        return tool_results, any_error

    def _call_claude(self, messages: List[Dict[str, Any]], system_content: str, tools: Optional[List] = None):
        """Build API params and make a single call to Claude."""
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        return self.client.messages.create(**api_params)

    @staticmethod
    def _extract_text(response) -> str:
        """Return the text of the first text content block, skipping thinking blocks."""
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""