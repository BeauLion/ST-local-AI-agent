// Wrapped in an IIFE, same reasoning as settings_panel.js - this file
// loads alongside prompt_log_viewer.js and settings_panel.js as plain
// scripts on dashboard.html, so top-level names must not collide.
(function () {

const statusEl = document.getElementById("modelStatus");
const infoEl = document.getElementById("modelInfo");
const presetSelect = document.getElementById("modelPresetSelect");
const repoInput = document.getElementById("modelRepoInput");
const nglInput = document.getElementById("modelNglInput");
const contextInput = document.getElementById("modelContextInput");
const swapBtn = document.getElementById("modelSwapBtn");
const resetBtn = document.getElementById("modelResetBtn");

let pollTimer = null;

function showStatus(msg, ok) {
  statusEl.textContent = msg;
  statusEl.hidden = false;
  statusEl.className = "status " + (ok ? "ok" : "err");
}

const PHASE_LABELS = {
  ready: "Ready", swapping: "Swapping…", starting: "Starting…",
  error: "Error", stopped: "Stopped",
};

function renderInfo(data) {
  infoEl.innerHTML = `
    <div class="row"><div class="label"><div class="name">Status</div></div><div>${PHASE_LABELS[data.phase] || data.phase}</div></div>
    <div class="row"><div class="label"><div class="name">Model</div></div><div>${data.repo}</div></div>
    <div class="row"><div class="label"><div class="name">GPU layers (ngl)</div></div><div>${data.ngl}</div></div>
    <div class="row"><div class="label"><div class="name">Context</div></div><div>${data.context}</div></div>
  `;

  if (data.last_error) {
    showStatus(data.last_error, data.phase === "ready");
  }

  const busy = data.phase === "swapping" || data.phase === "starting";
  swapBtn.disabled = busy;
  resetBtn.disabled = busy;

  if (busy && !pollTimer) {
    pollTimer = setInterval(refreshStatus, 2000);
  } else if (!busy && pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function refreshStatus() {
  try {
    const res = await fetch("/model/status");
    renderInfo(await res.json());
  } catch (e) {
    showStatus("Can't reach the agent server right now.", false);
  }
}

async function loadPresets() {
  const res = await fetch("/model/presets");
  const presets = await res.json();
  presetSelect.innerHTML = '<option value="">Custom (use fields below)</option>' +
    presets.map(p => `<option value="${p.label}">${p.label}</option>`).join("");
}

swapBtn.onclick = async () => {
  const body = presetSelect.value
    ? { preset: presetSelect.value }
    : {
        repo: repoInput.value.trim(),
        ngl: nglInput.value ? parseInt(nglInput.value, 10) : undefined,
        context: contextInput.value ? parseInt(contextInput.value, 10) : undefined,
      };

  if (!presetSelect.value && !body.repo) {
    showStatus("Pick a preset or enter a repo string.", false);
    return;
  }

  try {
    const res = await fetch("/model/swap", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error((await res.json()).detail || "Swap failed to start");
    showStatus("Swap started — this can take a minute or two. Chat will pause until it's done.", true);
    refreshStatus();
  } catch (e) {
    showStatus(e.message, false);
  }
};

resetBtn.onclick = async () => {
  try {
    const res = await fetch("/model/reset", { method: "POST" });
    if (!res.ok) throw new Error((await res.json()).detail || "Reset failed to start");
    showStatus("Resetting to config.py defaults…", true);
    refreshStatus();
  } catch (e) {
    showStatus(e.message, false);
  }
};

loadPresets();
refreshStatus();

})();
