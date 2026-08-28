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

Prompt-log integration: reuses main.py's own log_prompt()/log_console()
(same session log file, same viewer at /prompt-log-viewer) so a bad or
missing tool call here is exactly as inspectable as a main-agent turn -
the outgoing request, the model's raw response, and every alog() line
this module writes all land in one entry, tagged with the
"attire_subagent_*" section labels below. Always logged under iteration
0, since this is a single one-shot pass, not a multi-round loop - same
convention main.py's own MAX_TOOL_ITERATIONS bailout already uses for a
fixed, non-positional iteration number.

KNOWN EDGE CASE (accepted, not engineered around): because this writes
into the SAME session log file as the main agent, a sub-agent pass that
blows past its caller's timeout (see chat_completions) could still be
writing its own log entries after the NEXT turn's agent_loop has already
started writing its. The viewer pairs prompt/console entries positionally,
so a very late write could interleave with an unrelated turn's entries.
This only manifests when the 15s timeout is actually exceeded, which is
already the documented fail-open path elsewhere in this pipeline - not
solved here for the same reason it isn't solved there.

Fail-open by design throughout: any failure here is logged and swallowed,
never raised back into a real conversation turn. The worst case is one
turn of stale attire state, which is honest-if-late rather than wrong.
"""

import json

import httpx

import attire_manager
from config import AGENT_API_KEY, LLAMA_SERVER_URL
from console_log import alog
from prompt_log_engine import log_prompt, log_console

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
    "else in the slot is left alone automatically. Call this JUST AS "
    "READILY as attire_add_item - removals are just as common as "
    "additions and just as important to log, not an afterthought.\n"
    "- attire_replace_slot: ONLY for a genuine full change - a character "
    "changes into a whole new outfit, or the scene explicitly resets what "
    "someone is wearing. This wipes the slot, so never use it for a "
    "single item coming on or off.\n\n"
    "Worked example 1 (addition): current state shows feet: 'yellow "
    "socks'. The text says she puts on black shoes. Nothing said the "
    "socks came off, so this is a layering addition - call "
    "attire_add_item(slot='feet', item='black shoes'). Do NOT call "
    "attire_replace_slot here; that would silently delete the socks even "
    "though she's still wearing them.\n\n"
    "Worked example 2 (removal): current state shows feet: 'yellow "
    "socks, black shoes'. The text says he takes his shoes off. Call "
    "attire_remove_item(slot='feet', item_hint='black shoes'). The socks "
    "are untouched by this call. Do NOT skip this just because only one "
    "item came off, and do NOT call attire_replace_slot - that would "
    "also wipe the socks, which are still being worn.\n\n"
    "Call a tool once per change you're confident about - additions and "
    "removals both count and are equally worth logging. If nothing "
    "described a clothing change, call no tool at all - silence is the "
    "normal, expected outcome for most turns. Never narrate, never "
    "comment, never guess at a change that wasn't explicitly described."
)

_MUTATING_TOOLS = {"attire_add_item", "attire_remove_item", "attire_replace_slot"}

_SECTION_LABELS = [
    "attire_subagent_system",
    "attire_subagent_current_state",
    "attire_subagent_user_turn",
    "attire_subagent_assistant_turn",
]


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

    # iteration is always 0 - this is a single one-shot pass, not a
    # multi-round loop, so there's no meaningful iteration to track.
    log_prompt(body, 0, _SECTION_LABELS)

    try:
        async with httpx.AsyncClient(
            timeout=30.0, headers={"Authorization": f"Bearer {AGENT_API_KEY}"}
        ) as client:
            resp = await client.post(f"{LLAMA_SERVER_URL}/v1/chat/completions", json=body)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        alog(f"[ATTIRE-SUBAGENT] Request to llama-server failed: {e}")
        log_console(0, {"kind": "error", "text": str(e)})
        return

    choices = data.get("choices") or [{}]
    message = choices[0].get("message", {})
    # Only populated if llama-server is run with --reasoning-format and
    # the loaded model actually produces a reasoning block - same as
    # main.py's own agent_loop, just read from the non-streaming message
    # shape instead of accumulated across streamed deltas.
    thinking = message.get("reasoning_content") or None
    calls = message.get("tool_calls") or []

    if not calls:
        alog("[ATTIRE-SUBAGENT] No attire change detected this turn.")
        log_console(
            0,
            {"kind": "content", "text": message.get("content") or "(no attire change)"},
            thinking=thinking,
        )
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

    # Same kind/shape main.py's own agent_loop logs for a tool-calling
    # iteration - the raw model tool_calls payload as `text`; the
    # per-call outcomes above ride along as this entry's buffered
    # console `lines` via alog(), same pairing the main agent gets.
    log_console(0, {"kind": "tool_calls", "text": json.dumps(calls, indent=2)}, thinking=thinking)
