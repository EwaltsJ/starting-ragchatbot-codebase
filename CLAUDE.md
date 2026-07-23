# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Always use `uv` for dependency management and running the server — never `pip` or bare `python`/`uvicorn` directly.**

```bash
uv sync                       # install/sync dependencies
```

Run the app (must run from `backend/`, since it mounts `../frontend` and `../docs` as relative paths):

```bash
./run.sh                                              # from repo root
# or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

- Web UI: `http://localhost:8000`
- API docs (Swagger): `http://localhost:8000/docs`

Requires `ANTHROPIC_API_KEY` in a `.env` file at the repo root.

There are no tests, linter, or type-checker configured in this repo currently.

## Architecture

This is a RAG chatbot that answers questions about course transcripts stored in `docs/`, using ChromaDB for retrieval and Claude for generation, with a single tool call mediating between them.

**Request flow:** `frontend/script.js` → `POST /api/query` (`backend/app.py`) → `RAGSystem.query()` (`backend/rag_system.py`, the central orchestrator) → `AIGenerator` (`backend/ai_generator.py`) calls Claude with the `search_course_content` tool available.

- Claude decides per-query whether to search: general knowledge questions are answered directly; course-specific questions trigger exactly one tool call (enforced by the system prompt in `ai_generator.py`, not by code — "One search per query maximum").
- When a tool call happens, `ToolManager.execute_tool()` (`backend/search_tools.py`) invokes `CourseSearchTool`, which calls `VectorStore.search()` (`backend/vector_store.py`). Results go back to Claude for a second, final API call to synthesize the answer — so a single user query can mean 1 or 2 Claude API calls.
- Sources shown in the UI are a side channel: `CourseSearchTool` tracks `last_sources` after formatting search results, which `RAGSystem` pulls via `tool_manager.get_last_sources()` and resets after each query — they are not part of Claude's answer text.
- Conversation history is kept per-session in-memory (`SessionManager`, `backend/session_manager.py`), capped at `MAX_HISTORY` exchanges, and passed to Claude as a formatted string in the system prompt, not as multi-turn `messages`.

**Vector store layout (`backend/vector_store.py`):** ChromaDB has two collections:
- `course_catalog` — one entry per course (title as ID, plus instructor/link/lessons-as-JSON metadata). Used only for fuzzy course-name resolution: a search for `course_name="MCP"` runs a semantic query against this collection to resolve to the actual course title before filtering `course_content`.
- `course_content` — the chunked lesson text actually searched, filterable by `course_title` and/or `lesson_number`.

**Document ingestion (`backend/document_processor.py`):** On startup, `app.py`'s startup event loads every file in `docs/` via `RAGSystem.add_course_folder()`, skipping courses whose title already exists in the vector store (so re-running doesn't re-embed). Expected file format:

```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <lesson title>
Lesson Link: <url>
<lesson content...>

Lesson 1: <next lesson title>
<lesson content...>
```

Parsing splits on `Lesson N: ...` markers, then chunks each lesson's text into sentence-based, overlapping chunks (`CHUNK_SIZE`/`CHUNK_OVERLAP` in `config.py`). Note: the first-chunk-of-lesson prefixing logic differs between the "previous lesson" branch and the "last lesson" branch in `process_course_document` (one prepends `"Lesson N content: "` only to chunk 0, the other prepends `"Course {title} Lesson {N} content: "` to every chunk) — be aware of this inconsistency if touching chunk formatting.

**Config (`backend/config.py`):** single dataclass loaded from `.env`, covers the Anthropic model name, embedding model (`all-MiniLM-L6-v2`), chunk size/overlap, max search results, and max history — read this before changing retrieval or generation behavior.
