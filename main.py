"""
Phase 1: Agent server skeleton.

Right now this does nothing "smart" - it just forwards every request from
SillyTavern straight to llama-server and returns the answer unchanged.
This proves the plumbing works before we add tools, memory, and reasoning
in later phases.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI()

# Where llama-server is listening (from Phase 0)
LLAMA_SERVER_URL = "http://localhost:8080"


@app.get("/v1/models")
async def list_models():
    """SillyTavern calls this to populate the model dropdown."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{LLAMA_SERVER_URL}/v1/models")
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    The main endpoint SillyTavern talks to for every message.
    This is where Phases 2-4 will add tool calls, memory, and reasoning.
    For now: pure pass-through to llama-server.
    """
    body = await request.json()
    is_streaming = body.get("stream", False)

    if is_streaming:
        async def stream_response():
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST", f"{LLAMA_SERVER_URL}/v1/chat/completions", json=body
                ) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    else:
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(
                f"{LLAMA_SERVER_URL}/v1/chat/completions", json=body
            )
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
