"""
start.py — launches the agent server.

llama-server is now launched and owned by main.py itself (see
model_manager.py + its lifespan hook in main.py), not by this script -
that's what lets /model/swap stop and restart llama-server on request
without you touching a terminal. This script's job shrank to just:

  1. Launch the agent server (uvicorn) and wait for it to report ready
     (main.py's own lifespan blocks serving until llama-server's health
     check passes, so "ready" here already means the model is loaded)
  2. Stream its output into this one terminal
  3. On Ctrl+C, shut it down cleanly

Run with:
    python start.py

NOTE ON --reload: intentionally OFF below. With llama-server's lifecycle
now tied to main.py's own startup/shutdown, uvicorn's --reload would
restart llama-server (a 10-60+ second model load) every time you save a
change to any .py file in this folder while actively developing main.py -
not just main.py itself, since --reload watches the whole directory.
If you're doing heavy main.py development, run
`uvicorn main:app --host 0.0.0.0 --port 8100 --reload` directly instead
of this script for that session; use start.py for normal day-to-day use.
"""

import subprocess
import sys
import time
import signal
import os
import httpx

import config


def stream_output(proc: subprocess.Popen, label: str):
    for line in proc.stdout:
        print(f"[{label}] {line}", end="")


def start_agent_server() -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "uvicorn", "main:app",
        "--host", config.AGENT_SERVER_HOST,
        "--port", str(config.AGENT_SERVER_PORT),
    ]
    print(f"[start.py] Launching agent server (which will launch llama-server itself):\n  {' '.join(cmd)}\n")

    # Force unbuffered stdout on the child process. Without this, print()
    # statements in main.py (our [AGENT] logs) get block-buffered because
    # stdout is a pipe, not a terminal — they can sit invisible for a long
    # time while uvicorn's own logging-based access log lines (which flush
    # immediately) show up right away. This made it look like nothing was
    # happening when it actually was.
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Windows' default console/pipe encoding for a Python process is often
    # the legacy cp1252 codepage, which can't represent every Unicode
    # character (e.g. certain punctuation in a real iCloud calendar name,
    # or curly quotes/em-dashes in free text). Without this, any tool
    # result containing such a character crashes the debug print()
    # statements in main.py (and takes the whole request down with it) -
    # this forces UTF-8 instead, which can represent anything.
    env["PYTHONIOENCODING"] = "utf-8"

    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
        # PYTHONIOENCODING above makes main.py WRITE utf-8; this makes
        # start.py correctly READ it back on this end of the same pipe.
        # Both sides need to agree, or Windows falls back to decoding with
        # cp1252 here regardless of what was actually written.
        encoding="utf-8", errors="replace",
    )


def wait_for_agent_server(timeout_seconds: int = 180) -> bool:
    """
    Polls the agent server's own health, which only starts responding once
    its lifespan startup (llama-server launch + health check) has finished -
    so "ready" here means the whole stack is up, model included. Longer
    default timeout than the old llama-only wait, since it now also covers
    llama-server's own load time inside it.
    """
    url = f"http://localhost:{config.AGENT_SERVER_PORT}/model/status"
    print(f"[start.py] Waiting for the agent server (and llama-server) to become ready at {url} ...")
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            resp = httpx.get(url, timeout=2)
            if resp.status_code == 200 and resp.json().get("phase") == "ready":
                print("[start.py] Agent server and llama-server are both ready.\n")
                return True
        except httpx.RequestError:
            pass
        time.sleep(1)
    return False


def main():
    agent_proc = start_agent_server()

    def shutdown(signum=None, frame=None):
        print("\n[start.py] Shutting down...")
        agent_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    if not wait_for_agent_server():
        print("[start.py] WARNING: didn't confirm ready in time - check the output below for errors.")
        print("  (The agent server keeps running either way; this is just a startup confirmation check.)")

    try:
        stream_output(agent_proc, "agent")
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
