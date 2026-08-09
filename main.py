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
import tempfile
from datetime import datetime
from pathlib import Path

import docker

import httpx
from ddgs import DDGS
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

import memory

app = FastAPI()

LLAMA_SERVER_URL = "http://localhost:8080"
MAX_TOOL_ITERATIONS = 8  # raised from 5 to allow longer reasoning chains

# The ONLY folder the agent is allowed to read files from.
SAFE_FILES_DIR = (Path(__file__).parent / "agent_files").resolve()
SAFE_FILES_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling schema) + the Python functions
# that actually execute them. Add new tools here in later phases.
# ---------------------------------------------------------------------------

def get_current_time(args: dict) -> str:
    return datetime.now().strftime("%A, %Y-%m-%d %H:%M:%S")


def calculate(args: dict) -> str:
    expression = args.get("expression", "")
    if not expression:
        return "Error: no expression provided."

    # Only allow digits, arithmetic operators, parentheses, decimals, and
    # whitespace - blocks any attempt to run arbitrary Python via eval.
    allowed = set("0123456789.+-*/() \t")
    if not set(expression) <= allowed:
        return "Error: expression contains characters that aren't allowed (only numbers and + - * / ( ) are permitted)."

    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


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


def get_weather(args: dict) -> str:
    location = args.get("location", "")
    if not location:
        return "Error: no location provided."

    try:
        geo_resp = httpx.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1},
            timeout=10,
        )
        geo_results = geo_resp.json().get("results")
        if not geo_results:
            return f"Could not find a location matching '{location}'."

        place = geo_results[0]
        lat, lon = place["latitude"], place["longitude"]

        weather_resp = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,weather_code,wind_speed_10m",
            },
            timeout=10,
        )
        current = weather_resp.json().get("current", {})
        if "temperature_2m" not in current:
            return "Weather service did not return current data."

        name = place.get("name", location)
        country = place.get("country", "")
        return (
            f"Current weather in {name}, {country}: "
            f"{current['temperature_2m']}°C, "
            f"wind {current.get('wind_speed_10m')} km/h "
            f"(observed at {current.get('time')})"
        )
    except Exception as e:
        return f"Weather lookup failed: {e}"


def save_memory_tool(args: dict) -> str:
    fact = args.get("fact", "")
    if not fact:
        return "Error: no fact provided."
    memory.save_memory(fact)
    return f"Saved to long-term memory: {fact}"


def search_documents_tool(args: dict) -> str:
    query = args.get("query", "")
    if not query:
        return "Error: no query provided."
    results = memory.search_documents(query, SAFE_FILES_DIR)
    if not results:
        return "No relevant content found in the indexed documents."
    lines = []
    for source, text in results:
        snippet = text.strip().replace("\n", " ")[:400]
        lines.append(f"[{source}]: {snippet}")
    return "\n".join(lines)


_docker_client = None


def _get_docker_client():
    global _docker_client
    if _docker_client is None:
        _docker_client = docker.from_env()
    return _docker_client


def run_python(args: dict) -> str:
    code = args.get("code", "")
    if not code:
        return "Error: no code provided."

    try:
        client = _get_docker_client()
    except Exception as e:
        return f"Error: Docker isn't available ({e}). Is Docker Desktop running?"

    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = Path(tmp_dir) / "snippet.py"
        script_path.write_text(code, encoding="utf-8")

        container = None
        try:
            container = client.containers.run(
                "python:3.12-slim",
                command=["python", "/sandbox/snippet.py"],
                volumes={tmp_dir: {"bind": "/sandbox", "mode": "ro"}},
                working_dir="/sandbox",
                network_disabled=True,   # no internet access from inside
                mem_limit="256m",
                nano_cpus=1_000_000_000,  # capped at 1 CPU core
                detach=True,
            )
            result = container.wait(timeout=10)
            exit_code = result.get("StatusCode", 1)
            logs = container.logs().decode("utf-8", errors="replace")[-3000:]
        except docker.errors.ImageNotFound:
            return "Error: python:3.12-slim image not found. Run 'docker pull python:3.12-slim' once."
        except Exception as e:
            return f"Error running sandboxed code: {e}"
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

        if exit_code != 0:
            return f"Exit code {exit_code}. Output:\n{logs}"
        return logs or "(no output, nothing was printed)"


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
            "name": "calculate",
            "description": "Evaluate an exact arithmetic expression (numbers and + - * / ( ) only). Always use this for any math instead of computing it yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '(2026 - 1889)'"}
                },
                "required": ["expression"],
            },
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
            "name": "get_weather",
            "description": "Get the current real-world weather and temperature for a specific place. Always use this instead of web_search for weather/temperature questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and country, e.g. 'Amsterdam, Netherlands'."}
                },
                "required": ["location"],
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
            "name": "save_memory",
            "description": "Save an important fact about the user for recall in future conversations (e.g. their name, preferences, ongoing projects). Do not save trivial small talk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "The fact to remember, written as a standalone sentence."}
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Semantically search the user's uploaded .txt documents for relevant passages. Use this to find specific information within long or multiple documents, instead of read_file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run a short Python snippet for calculations, data processing, or logic too complex for the calculate tool. Executes in an isolated Docker container with no network access and a 10-second timeout. Use print() for output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code. Use print() to produce output."}
                },
                "required": ["code"],
            },
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
    "calculate": calculate,
    "run_python": run_python,
    "web_search": web_search,
    "get_weather": get_weather,
    "list_files": list_files,
    "read_file": read_file,
    "save_memory": save_memory_tool,
    "search_documents": search_documents_tool,
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

    tool_instruction = {
        "role": "system",
        "content": (
            "You have tools available: get_current_time, calculate, run_python, "
            "web_search, get_weather, list_files, read_file, save_memory, "
            "search_documents. You MUST call the relevant tool whenever the user "
            "asks about current events, real-time facts, dates/times, weather, "
            "exact arithmetic, or anything you are not fully certain of from "
            "memory. For weather/temperature questions, always use get_weather, "
            "never web_search. Use search_documents (not read_file) when looking "
            "for specific information inside long or multiple documents. Use "
            "calculate for simple arithmetic, or run_python for anything needing "
            "actual code logic. Call save_memory when the user shares a durable "
            "fact about themselves worth remembering - not for small talk. "
            "IMPORTANT for multi-step questions: if answering fully requires "
            "several pieces of information, call tools one at a time in sequence, "
            "using each result to decide your next step, before giving your final "
            "answer. Do not stop after one tool call if the question isn't fully "
            "answered yet. Never guess or invent facts, dates, statistics, or "
            "search results that a tool could actually check for you. If a tool "
            "returns no useful result, say so honestly instead of making "
            "something up."
        ),
    }
    messages_to_prepend = [tool_instruction]

    # Auto-recall: silently check if any saved memories are relevant to what
    # the user just said, and inject them - no tool call needed for this part.
    last_user_msg = next(
        (m["content"] for m in reversed(body["messages"]) if m.get("role") == "user"),
        None,
    )
    if last_user_msg:
        relevant = await asyncio.to_thread(memory.search_memories, last_user_msg)
        if relevant:
            print(f"[AGENT] Recalled {len(relevant)} relevant memory item(s)")
            messages_to_prepend.append({
                "role": "system",
                "content": "Relevant things you remember about this user from past "
                            "conversations:\n" + "\n".join(f"- {m}" for m in relevant),
            })

    upstream_body["messages"] = messages_to_prepend + upstream_body["messages"]

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
                print("[AGENT] Model answered directly, without calling any tool.")
                final_data = data
                break

            print(f"[AGENT] Model requested {len(tool_calls)} tool call(s)")

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

                print(f"[AGENT] {name}({args}) ->")
                print(f"[AGENT]   {str(result)[:400]}")

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
