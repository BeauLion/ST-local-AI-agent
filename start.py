"""
start.py — one script to launch everything.

Run this instead of typing two separate commands in two terminals:
    python start.py

It will:
  1. Launch llama-server using the command built from config.py
  2. Wait until llama-server responds to health checks (so the agent
     server never starts before the model is actually ready)
  3. Launch the agent server (uvicorn)
  4. Stream both processes' output into this one terminal
  5. On Ctrl+C, shut both processes down cleanly

If either process fails to start, this script will tell you which one
and why, instead of leaving you guessing.
"""

import subprocess
import sys
import time
import signal
import threading
import os
import httpx

import config


def stream_output(proc: subprocess.Popen, label: str):
    for line in proc.stdout:
        print(f"[{label}] {line}", end="")


def start_llama_server() -> subprocess.Popen:
    cmd = config.build_llama_server_command()
    print(f"[start.py] Launching llama-server:\n  {' '.join(cmd)}\n")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        print(f"[start.py] ERROR: could not find llama-server.exe at:")
        print(f"  {config.LLAMA_SERVER_EXE}")
        print("  Check LLAMA_SERVER_EXE in config.py.")
        sys.exit(1)

    # Read llama-server's output continuously on a background thread.
    # Without this, its stdout pipe buffer fills up once it writes enough
    # startup text and the process silently hangs, never finishing boot.
    threading.Thread(
        target=stream_output, args=(proc, "llama"), daemon=True
    ).start()

    return proc


def wait_for_llama_server(timeout_seconds: int = 120) -> bool:
    """Polls llama-server until it responds or we give up."""
    url = f"{config.LLAMA_SERVER_URL}/health"
    print(f"[start.py] Waiting for llama-server to become ready at {url} ...")
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            resp = httpx.get(url, timeout=2)
            if resp.status_code == 200:
                print("[start.py] llama-server is ready.\n")
                return True
        except httpx.RequestError:
            pass
        time.sleep(1)
    return False


def start_agent_server() -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "uvicorn", "main:app",
        "--host", config.AGENT_SERVER_HOST,
        "--port", str(config.AGENT_SERVER_PORT),
        "--reload",
    ]
    print(f"[start.py] Launching agent server:\n  {' '.join(cmd)}\n")

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
    )


def main():
    llama_proc = start_llama_server()

    if not wait_for_llama_server():
        print("[start.py] ERROR: llama-server did not become ready in time.")
        print("  Check the output above for errors (e.g. model download, VRAM issues).")
        llama_proc.terminate()
        sys.exit(1)

    agent_proc = start_agent_server()

    def shutdown(signum=None, frame=None):
        print("\n[start.py] Shutting down...")
        agent_proc.terminate()
        llama_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    # Stream agent server output on the main thread. llama-server's own
    # output has already been flowing since it started; we don't need to
    # interleave it further, agent server logs are what you'll watch most.
    try:
        stream_output(agent_proc, "agent")
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()