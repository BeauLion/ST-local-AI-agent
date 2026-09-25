"""
model_engine.py — the /model API for swapping the running llama-server
model without a manual restart. Mirrors settings_engine.py's pattern.

  GET  /model/status   -> {phase, repo, ngl, context, md, spec_type, last_error, started_at}
  GET  /model/presets  -> config.MODEL_PRESETS (list, may be empty)
  POST /model/swap      body: {"preset": "label"}  OR
                               {"repo": "...", "ngl": 99, "context": 32768}
                         (ngl/context/md/spec_type optional - omitted ones
                         keep whatever's currently running)
  POST /model/reset    -> swap back to config.py's own LLAMA_* defaults
                          (LLAMA_SWAP_DEFAULT_MODEL on the llama-swap backend)

On the llama-swap backend (config.LLAMA_BACKEND), /model/presets lists the
models configured in llama-swap itself instead of config.MODEL_PRESETS, and
a swap only needs "repo" (= the llama-swap model name).

Swap and reset both return immediately with {"status": "swapping"} - the
actual stop/start/health-check happens in a background task. Poll
/model/status (the settings panel does this every 2s) to see when it's
done, and whether it landed on the requested model or rolled back.
"""

import asyncio
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

import config
import model_manager

router = APIRouter()


@router.get("/model_panel.js")
async def model_panel_js():
    js_path = Path(__file__).parent / "web" / "model_panel.js"
    return Response(content=js_path.read_text(encoding="utf-8"), media_type="application/javascript")

# Guards against two swaps racing each other. model_manager.swap() also
# checks its own "swapping" phase, but that check happens INSIDE the
# background thread - this lock rejects a second request immediately,
# before even spawning a redundant thread.
_swap_lock = asyncio.Lock()


@router.get("/model/status")
async def model_status():
    return model_manager.status()


@router.get("/model/presets")
async def model_presets():
    if config.LLAMA_BACKEND == "llama-swap":
        try:
            names = await asyncio.to_thread(model_manager.list_remote_models)
        except (httpx.HTTPError, ValueError, KeyError):
            return []
        return [{"label": name, "repo": name} for name in names]
    return getattr(config, "MODEL_PRESETS", [])


def _run_in_background(fn, **kwargs):
    async def _run():
        async with _swap_lock:
            await asyncio.to_thread(fn, **kwargs)
    asyncio.create_task(_run())


@router.post("/model/swap")
async def model_swap(request: Request):
    if _swap_lock.locked():
        raise HTTPException(409, "A model swap is already in progress.")

    body = await request.json()

    if "preset" in body:
        presets = await model_presets()
        match = next((p for p in presets if p.get("label") == body["preset"]), None)
        if match is None:
            raise HTTPException(400, f"Unknown preset: {body['preset']}")
        kwargs = {
            "repo": match.get("repo"),
            "ngl": match.get("ngl"),
            "context": match.get("context"),
            "md": match.get("md"),
            "spec_type": match.get("spec_type"),
        }
    else:
        if not body.get("repo"):
            raise HTTPException(400, 'Body must include either {"preset": "..."} or a "repo"')
        kwargs = {
            "repo": body.get("repo"),
            "ngl": body.get("ngl"),
            "context": body.get("context"),
            "md": body.get("md"),
            "spec_type": body.get("spec_type"),
        }

    _run_in_background(model_manager.swap, **kwargs)
    return {"status": "swapping", "target": kwargs}


@router.post("/model/reset")
async def model_reset():
    if _swap_lock.locked():
        raise HTTPException(409, "A model swap is already in progress.")

    _run_in_background(model_manager.reset)
    return {"status": "swapping", "target": "default model"}
