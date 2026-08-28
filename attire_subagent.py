"""
Post-turn attire sub-agent: a separate, one-shot completion call against the
same llama-server, run after every finished turn and decoupled entirely from
the main agent's own tool selection.

Attire changes are easy for the main agent to skip mid-narration since
they're rarely the user's actual request - this pass exists purely as a
second look at the exchange that just completed, with nothing to do BUT
notice attire changes and log them. It re-sends only the last user/assistant
exchange plus the attire tools (see main.py's ATTIRE_TOOL_SCHEMAS - reused
here rather than hand-copied so the two can't silently drift apart), and
executes whatever attire_manager_update call the model makes directly
against attire_manager.py.

See config.py's ATTIRE_SUBAGENT_TIMEOUT_SECONDS and main.py's
chat_completions/_spawn_attire_subagent for how this fits into the turn
cycle: it's spawned fire-and-forget when a turn finishes, and the START of
the NEXT turn waits on it (with a timeout) before reading attire state for
context injection.
"""

import json

import httpx

import attire_manager
from attire_manager import AttireManagerError
from config import AGENT_API_KEY, LLAMA_SERVER_URL
from console_log import alog

_SYSTEM_PROMPT = (
    "You are a narrow attire-tracking pass reviewing one exchange for any change "
    "in what a character is wearing. Call attire_manager_update once per character "
    "whose attire changed - full outfit changes as well as subtle or partial ones "
    "(loosening or removing a tie, unbuttoning a shirt, taking off shoes or an "
    "accessory, a jacket coming off, one item swapped for another). Only pass the "
    "slot(s) that actually changed. If nothing about anyone's attire changed in "
    "this exchange, respond with no tool calls at all."
)


def _execute_attire_tool_call(call: dict) -> str:
    name = call.get("function", {}).get("name")
    try:
        args = json.loads(call.get("function", {}).get("arguments") or "{}")
    except json.JSONDecodeError:
        return "Error: malformed tool arguments."

    if name == "attire_manager_update":
        try:
            record, changed = attire_manager.update_attire(
                args.get("character_name", ""),
                head=args.get("head"),
                top=args.get("top"),
                bottom=args.get("bottom"),
                feet=args.get("feet"),
                accessories=args.get("accessories"),
            )
            if not changed:
                return f"No change needed - {record['name']}'s attire already matches that."
            return f"Updated {record['name']}'s attire ({', '.join(changed)})."
        except AttireManagerError as e:
            return f"Error: {e}"

    if name == "attire_manager_get":
        return attire_manager.get_attire_text(args.get("character_name", ""))

    return f"Error: unknown tool {name!r}"


async def run_attire_subagent(user_text: str, assistant_text: str) -> None:
    """Fire-and-forget: asks the model, via a fresh completion request
    carrying only the attire tools, whether the exchange just finished
    changed anyone's attire, and applies any resulting attire_manager_update
    call. Never raises - by the time this runs the user-facing turn has
    already completed, so a failure here should just mean stale attire
    state, never a chat-facing error."""
    import main  # deferred: main.py imports this module at startup, so
    # ATTIRE_TOOL_SCHEMAS only exists on main by the time this function
    # actually runs (never at attire_subagent's own import time).

    if not user_text and not assistant_text:
        return

    body = {
        "model": "agent",
        "stream": False,
        "tools": main.ATTIRE_TOOL_SCHEMAS,
        "tool_choice": "auto",
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
    }

    try:
        async with httpx.AsyncClient(
            timeout=60.0, headers={"Authorization": f"Bearer {AGENT_API_KEY}"}
        ) as client:
            resp = await client.post(f"{LLAMA_SERVER_URL}/v1/chat/completions", json=body)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        alog(f"[ATTIRE-SUBAGENT] Request to llama-server failed: {e}")
        return

    message = (data.get("choices") or [{}])[0].get("message") or {}
    calls = message.get("tool_calls") or []
    if not calls:
        return

    for call in calls:
        function = call.get("function", {})
        try:
            result = _execute_attire_tool_call(call)
            alog(f"[ATTIRE-SUBAGENT] {function.get('name')}({function.get('arguments')}) -> {result}")
        except Exception as e:
            alog(f"[ATTIRE-SUBAGENT] Tool execution failed: {e}")
