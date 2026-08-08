"""
Phase 2: Tool-calling loop.

The agent server now:
  1. Injects a list of available tools into every request sent to llama-server
  2. If the model asks to call a tool, executes it in Python and feeds the
     result back to the model
  3. Loops until the model gives a final answer (or we hit a safety limit)
  4. Returns that final answer to SillyTavern, in whatever format
     (streaming or not) SillyTavern originally asked for

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8100 --reload
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import httpx
from ddgs import DDGS
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI()

LLAMA_SERVER_URL = "http://localhost:8080"
MAX_TOOL_ITERATIONS = 5  # safety cap so a confused model can't loop forever

# The ONLY folder the agent is allowed to read files from.
SAFE_FILES_DIR = (Path(__file__).parent / "agent_files").resolve()
SAFE_FILES_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling schema) + the Python functions
# that actually execute them. Add new tools here in later phases.
# ---------------------------------------------------------------------------

def get_current_time(args: dict) -> str:
    return datetime.now().strftime("%A, %Y-%m-%d %H:%M:%S")


def web_search(args: dict) -> str:
    query = args.get("query", "")
    if not query:
        return "Error: no query provided."
    try:
        results = DDGS().text(query, max_results=5)
    except Exception as e:
        return f"Search failed: {e}"

    if not results:
        return "No results found."

    lines = []
    for r in results:
        lines.append(f"- {r['title']}\n  {r['href']}\n  {r['body']}")
    return "\n".join(lines)


def list_files(args: dict) -> str:
    files = [f.name for f in SAFE_FILES_DIR.iterdir() if f.is_file()]
    if not files:
        return f"No files in {SAFE_FILES_DIR}."
    return "\n".join(files)


def read_file(args: dict) -> str:
    filename = args.get("filename", "")
    if not filename:
        return "Error: no filename provided."

    # Resolve the path and make sure it's still inside SAFE_FILES_DIR.
    # This blocks tricks like "../../secrets.txt".
    target = (SAFE_FILES_DIR / filename).resolve()
    if SAFE_FILES_DIR not in target.parents and target != SAFE_FILES_DIR:
        return "Error: access denied outside the allowed folder."
    if not target.is_file():
        return f"Error: '{filename}' not found."

    try:
        return target.read_text(encoding="utf-8", errors="replace")[:5000]
    except Exception as e:
        return f"Error reading file: {e}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current real-world date and time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information not in your training data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the files available for reading.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file from the allowed files folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Name of the file to read."}
                },
                "required": ["filename"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
    "web_search": web_search,
    "list_files": list_files,
    "read_file": read_file,
}


# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models():
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{LLAMA_SERVER_URL}/v1/models")
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    client_wants_stream = body.get("stream", False)

    # We handle streaming ourselves at the end, after the tool loop is done.
    # This makes the loop logic far simpler.
    upstream_body = dict(body)
    upstream_body["stream"] = False
    upstream_body["tools"] = TOOLS
    upstream_body["tool_choice"] = "auto"

    final_data = None

    async with httpx.AsyncClient(timeout=None) as client:
        for _ in range(MAX_TOOL_ITERATIONS):
            resp = await client.post(
                f"{LLAMA_SERVER_URL}/v1/chat/completions", json=upstream_body
            )
            data = resp.json()
            message = data["choices"][0]["message"]

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                # No tool requested -> this is the final answer.
                final_data = data
                break

            # The model wants to call one or more tools.
            # 1. Add its tool-call message to the conversation.
            upstream_body["messages"].append(message)

            # 2. Execute each requested tool and add the result.
            for call in tool_calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}

                if name in TOOL_FUNCTIONS:
                    # Run in a thread so a slow web search doesn't freeze the server.
                    result = await asyncio.to_thread(TOOL_FUNCTIONS[name], args)
                else:
                    result = f"Error: unknown tool '{name}'"

                upstream_body["messages"].append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": str(result),
                    }
                )
            # loop again so the model can use the tool result

        if final_data is None:
            # Hit MAX_TOOL_ITERATIONS without a final answer - bail out safely.
            final_data = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "(Agent stopped: too many tool calls in a row.)",
                        },
                        "finish_reason": "stop",
                    }
                ]
            }

    final_message = final_data["choices"][0]["message"]

    if not client_wants_stream:
        return JSONResponse(content=final_data)

    # Client wanted streaming - fake a single-chunk SSE stream so
    # SillyTavern (which expects text/event-stream) still parses it correctly.
    async def fake_stream():
        chunk = {
            "choices": [
                {
                    "delta": {"content": final_message.get("content", "")},
                    "finish_reason": None,
                }
            ]
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode()

        done_chunk = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(done_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(fake_stream(), media_type="text/event-stream")
