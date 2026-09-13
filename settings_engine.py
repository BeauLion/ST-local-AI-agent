"""
settings_engine.py — the /settings API and the settings_panel.html page
that talks to it, isolated out of main.py the same way prompt_log_engine.py
isolates the prompt-log viewer.

main.py wires this in with:

    from settings_engine import router as settings_router
    app.include_router(settings_router)

Endpoints:
  GET  /settings              -> {name: {value, default, overridden, ...}}
  POST /settings               body: {"NAME": value, ...}  (one or many)
  POST /settings/reset         body: {"name": "NAME"}       (one setting)
  POST /settings/reset-all
  GET  /settings-panel         -> the HTML page
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, Response

import runtime_settings

router = APIRouter()


@router.get("/settings")
async def list_settings():
    return runtime_settings.get_all()


@router.post("/settings")
async def update_settings(request: Request):
    body = await request.json()
    if not isinstance(body, dict) or not body:
        raise HTTPException(400, "Body must be a non-empty JSON object of {setting_name: value}")
    try:
        return runtime_settings.set_many(body)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))


@router.post("/settings/reset")
async def reset_setting(request: Request):
    body = await request.json()
    name = body.get("name")
    if not name:
        raise HTTPException(400, "Body must include {\"name\": \"SETTING_NAME\"}")
    if name not in runtime_settings.SCHEMA:
        raise HTTPException(400, f"Unknown setting: {name}")
    return runtime_settings.reset(name)


@router.post("/settings/reset-all")
async def reset_all_settings():
    return runtime_settings.reset_all()


@router.get("/settings-panel")
async def settings_panel():
    html_path = Path(__file__).parent / "web" / "settings_panel.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@router.get("/settings_panel.css")
async def settings_panel_css():
    css_path = Path(__file__).parent / "web" / "settings_panel.css"
    return Response(content=css_path.read_text(encoding="utf-8"), media_type="text/css")


@router.get("/settings_panel.js")
async def settings_panel_js():
    js_path = Path(__file__).parent / "web" / "settings_panel.js"
    return Response(content=js_path.read_text(encoding="utf-8"), media_type="application/javascript")
