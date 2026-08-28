"""
Post-turn attire sub-agent: a narrow, one-shot background pass that scans
the most recently completed turn and updates attire_manager's authoritative
state - decoupled from the main agent's own tool-calling loop entirely.

Why this exists: the main agent already had attire_manager_update available
as a tool mid-conversation, but relying on it to notice every clothing
change WHILE also roleplaying and juggling other tools is exactly the kind
of split attention that gets missed. This module runs as a separate,
single-purpose completion call against the same llama-server, focused on
nothing but "did attire change, and if so, log it" - after the user-visible
turn is already done and sent.

Timing model (see main.py's chat_completions): fire-and-forget the moment
the main turn's final answer is known (kind == "done", never "handoff" -
a handoff isn't a finished turn). The NEXT incoming request awaits this
task, with a timeout, before it reads attire_manager's state for
injection - so the common case (user takes more than a few seconds to
read/type) costs nothing extra, and the worst case is a bounded wait,
never an indefinite hang.

Deliberately NOT using agent_loop: that machinery exists for the main
agent's multi-round tool chaining mid-conversation. This pass is one-shot -
a single non-streaming completion request, checked once for tool_calls -
because a turn's worth of attire changes needs at most a few direct calls,
not a back-and-forth loop.

v2 (see brainstorm-layered-clothing.md): the tool interface is three verbs
(add_item / remove_item / replace_slot) instead of one full-value
update_attire, specifically so a small model layering one item over
another ("shoes on over socks") can't accidentally erase the other by
getting a "restate the full value" instruction wrong - the add path
structurally cannot touch anything else in the slot.

Fail-open by design throughout: any failure here is logged and swallowed,
never raised back into a real conversation turn. The worst case is one
turn of stale attire state, which is honest-if-late rather than wrong.
"""

import json

import httpx

import attire_manager
from config import AGENT_API_KEY, LLAMA_SERVER_URL
from console_log import alog

_SYSTEM_PROMPT = (
    "You are a silent continuity tracker, not a conversational participant. "
    "You will be shown the currently recorded attire for every tracked "
    "character, followed by the most recent user message and the most "
    "recent assistant reply. You have three tools for logging a change - "
    "pick the one that matches what actually happened:\n\n"
    "- attire_add_item: something was put on, layered, or added, WITHOUT "
    "anything else coming off. Shoes going on over socks, a jacket going "
    "on over a shirt, a ring being put on - all additions. This is the "
    "right call any time you're unsure whether something underneath is "
    "still being worn; it never touches anything else already in the slot.\n"
    "- attire_remove_item: one specific item came off - taken off, "
    "removed, shrugged off, kicked off. Describe just that item; anything "
    "else in the slot is left alone automatically.\n"
    "- attire_replace_slot: ONLY for a genuine full change - a character "
    "changes into a whole new outfit, or the scene explicitly resets what "
    "someone is wearing. This wipes the slot, so never use it for a "
    "single item coming on or off.\n\n"
    "Worked example: current state shows feet: 'yellow socks'. The text "
    "says she puts on black shoes. Nothing said the socks came off, so "
    "this is a layering addition - call attire_add_item(slot='feet', "
    "item='black shoes'). Do NOT call attire_replace_slot here; that "
    "would silently delete the socks even though she's still wearing "
    "them.\n\n"
    "Call a tool once per change you're confident about. If nothing "
    "described a clothing change, call no tool at all - silence is the "
    "normal, expected outcome for most turns. Never narrate, never "
    "comment, never guess at a change that wasn't explicitly described."
)

_MUTATING_TOOLS = {"attire_add_item", "attire_remove_item", "attire_replace_slot"}


def _build_state_summary() -> str:
    """Every tracked character's current slots, reusing attire_manager's
    own formatting so this never drifts from what get_attire_text() shows
    the main agent."""
    state = attire_manager._load()
    if not state["characters"]:
        return "No characters are currently tracked."
    blocks = [
        f"{record['name']}:\n" + "\n".join(attire_manager._format_slots(record["slots"]))
        for record in state["characters"].values()
    ]
    return "\n\n".join(blocks)


async def run_attire_subagent(user_text: str, assistant_text: str) -> None:
    """One-shot: ask the model whether attire changed in the given
    exchange, apply any resulting tool calls directly against
    attire_manager. Never raises."""
    if not (user_text or "").strip() and not (assistant_text or "").strip():
        return

    # Deferred import: main.py imports this module to spawn the task, and
    # this module needs main's tool schemas/wrapper functions. Importing
    # main only here (not at module load time) avoids a circular import
    # between the two files - by the time this function actually runs,
    # main.py has long since finished loading.
    import main as agent_main

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "system", "content": f"[CURRENT ATTIRE STATE]\n{_build_state_summary()}"},
        {"role": "user", "content": user_text or ""},
        {"role": "assistant", "content": assistant_text or ""},
    ]
    body = {
        "messages": messages,
        "tools": agent_main.ATTIRE_TOOL_SCHEMAS,
        "tool_choice": "auto",
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(
            timeout=30.0, headers={"Authorization": f"Bearer {AGENT_API_KEY}"}
        ) as client:
            resp = await client.post(f"{LLAMA_SERVER_URL}/v1/chat/completions", json=body)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        alog(f"[ATTIRE-SUBAGENT] Request to llama-server failed: {e}")
        return

    choices = data.get("choices") or [{}]
    message = choices[0].get("message", {})
    calls = message.get("tool_calls") or []
    if not calls:
        alog("[ATTIRE-SUBAGENT] No attire change detected this turn.")
        return

    dispatch = {
        "attire_add_item": agent_main.attire_manager_add_item,
        "attire_remove_item": agent_main.attire_manager_remove_item,
        "attire_replace_slot": agent_main.attire_manager_replace_slot,
    }

    for call in calls:
        fn = call.get("function", {})
        name = fn.get("name")
        if name not in _MUTATING_TOOLS:
            # attire_manager_get is a read-only lookup - nothing to apply,
            # and anything else isn't a tool this sub-agent was given.
            alog(f"[ATTIRE-SUBAGENT] Ignoring non-mutating tool call: {name}")
            continue
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError as e:
            alog(f"[ATTIRE-SUBAGENT] Malformed tool arguments for {name}: {e}")
            continue
        result = dispatch[name](args)
        alog(f"[ATTIRE-SUBAGENT] {name}({args}) -> {result}")
