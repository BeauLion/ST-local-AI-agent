"""
config.py — single source of truth for every tunable value in the project.

Change a setting HERE, then restart start.py. Nothing else in the codebase
should contain a hardcoded port, path, threshold, or launch flag anymore —
if you find one, it belongs in this file instead.
"""

import os

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

# Root of this repo (folder this config.py file lives in). Everything else
# below is built relative to this, so the project stays portable.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Full path to your llama-server.exe (llama.cpp build). Update this if you
# ever move or reinstall llama.cpp.
LLAMA_SERVER_EXE = os.path.join(PROJECT_ROOT, "llama.cpp", "llama-server.exe")

# Sandboxed folder the write_file/read_file/list_files/search_documents
# tools are restricted to. Relative to PROJECT_ROOT.
SAFE_FILES_DIR = os.path.join(PROJECT_ROOT, "agent_files")

# Where memory.py stores its JSON files (memories.json, doc_index.json).
MEMORY_DATA_DIR = os.path.join(PROJECT_ROOT, "memory_data")


# ─────────────────────────────────────────────────────────────
# llama-server (the local inference engine, port 8080)
# ─────────────────────────────────────────────────────────────

LLAMA_SERVER_HOST = "0.0.0.0"
LLAMA_SERVER_PORT = 8080
LLAMA_SERVER_URL = f"http://localhost:{LLAMA_SERVER_PORT}"

# The exact model to pull/run via llama.cpp's -hf shorthand.
LLAMA_MODEL_REPO = "Qwen/Qwen2.5-14B-Instruct-GGUF:Q4_K_M"

LLAMA_NGL = 99          # -ngl: layers offloaded to GPU (99 = full offload)
LLAMA_CONTEXT = 8192    # -c: context window size
LLAMA_TEMP = 0.5        # --temp: lowered from default: fixed the malformed
                        #   tool-call bug (handover item 15). Raise this
                        #   again only if you're prepared to re-test that
                        #   fix, or re-add the salvage function.
LLAMA_FLASH_ATTENTION = 'on'   # -fa: flash attention, speed optimization [on|off|auto]
LLAMA_USE_JINJA = True         # --jinja: required for structured tool-call output


def build_llama_server_command() -> list[str]:
    """
    Builds the full llama-server launch command as a list of arguments,
    ready to pass to subprocess.Popen(). Mirrors exactly what you'd type
    by hand in PowerShell, just assembled from the settings above.
    """
    cmd = [
        LLAMA_SERVER_EXE,
        "-hf", LLAMA_MODEL_REPO,
        "-ngl", str(LLAMA_NGL),
        "-c", str(LLAMA_CONTEXT),
        "--temp", str(LLAMA_TEMP),
        "--host", LLAMA_SERVER_HOST,
        "--port", str(LLAMA_SERVER_PORT),
        "-fa", str(LLAMA_FLASH_ATTENTION),
    ]
    if LLAMA_USE_JINJA:
        cmd.append("--jinja")
    return cmd


# ─────────────────────────────────────────────────────────────
# Agent server (FastAPI app, port 8100 — talks to SillyTavern)
# ─────────────────────────────────────────────────────────────

AGENT_SERVER_HOST = "0.0.0.0"
AGENT_SERVER_PORT = 8100

# Ceiling on how many tool-call round-trips the agent will do before
# forcing a final answer, to prevent infinite tool-calling loops.
MAX_TOOL_ITERATIONS = 8


# ─────────────────────────────────────────────────────────────
# Memory / RAG (memory.py)
# ─────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Minimum cosine-similarity score for a saved memory to be considered
# relevant and injected into context. Lower = more permissive recall.
MEMORY_SIMILARITY_THRESHOLD = 0.15

# Stricter threshold for document RAG (search_documents), since documents
# are larger/noisier than short personal-memory facts.
DOCUMENT_SIMILARITY_THRESHOLD = 0.3


# ─────────────────────────────────────────────────────────────
# write_file tool
# ─────────────────────────────────────────────────────────────

WRITE_FILE_ALLOWED_EXTENSIONS = (".txt", ".md")
WRITE_FILE_MAX_CHARS = 20_000


# ─────────────────────────────────────────────────────────────
# run_python tool (Docker sandbox)
# ─────────────────────────────────────────────────────────────

DOCKER_IMAGE = "python:3.12-slim"
DOCKER_NETWORK_DISABLED = True
DOCKER_MEM_LIMIT = "256m"
DOCKER_CPU_COUNT = 1
DOCKER_TIMEOUT_SECONDS = 10