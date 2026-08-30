const groupsEl = document.getElementById("groups");
const statusEl = document.getElementById("status");
const resetAllBtn = document.getElementById("reset-all");

function showStatus(msg, ok) {
  statusEl.textContent = msg;
  statusEl.hidden = false;
  statusEl.className = "status " + (ok ? "ok" : "err");
  clearTimeout(showStatus._t);
  showStatus._t = setTimeout(() => { statusEl.hidden = true; }, 2500);
}

async function loadSettings() {
  const res = await fetch("/settings");
  const data = await res.json();
  render(data);
}

function render(data) {
  const groups = {};
  for (const [name, meta] of Object.entries(data)) {
    (groups[meta.group] ||= []).push({ name, ...meta });
  }

  groupsEl.innerHTML = "";
  for (const [groupName, items] of Object.entries(groups)) {
    const groupEl = document.createElement("div");
    groupEl.className = "group";
    const h2 = document.createElement("h2");
    h2.textContent = groupName;
    groupEl.appendChild(h2);

    for (const item of items) {
      groupEl.appendChild(renderRow(item));
    }
    groupsEl.appendChild(groupEl);
  }
}

function renderRow(item) {
  const row = document.createElement("div");
  row.className = "row";

  const label = document.createElement("div");
  label.className = "label";
  const nameEl = document.createElement("div");
  nameEl.className = "name";
  nameEl.textContent = item.label;
  if (item.overridden) {
    const badge = document.createElement("span");
    badge.className = "overridden-badge";
    badge.textContent = "modified";
    nameEl.appendChild(badge);
  }
  label.appendChild(nameEl);
  if (item.help) {
    const help = document.createElement("div");
    help.className = "help";
    help.textContent = item.help;
    label.appendChild(help);
  }
  row.appendChild(label);

  let control;
  if (item.type === "bool") {
    control = document.createElement("label");
    control.className = "switch";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = item.value;
    input.onchange = () => save(item.name, input.checked);
    const slider = document.createElement("span");
    slider.className = "slider";
    control.appendChild(input);
    control.appendChild(slider);
  } else {
    control = document.createElement("input");
    control.type = "number";
    control.step = item.type === "int" ? "1" : "0.01";
    control.min = item.min;
    control.max = item.max;
    control.value = item.value;
    control.onchange = () => {
      const v = item.type === "int" ? parseInt(control.value, 10) : parseFloat(control.value);
      save(item.name, v);
    };
  }
  row.appendChild(control);

  if (item.overridden) {
    const resetBtn = document.createElement("button");
    resetBtn.className = "reset-btn";
    resetBtn.textContent = "reset";
    resetBtn.onclick = () => resetOne(item.name);
    row.appendChild(resetBtn);
  }

  return row;
}

async function save(name, value) {
  try {
    const res = await fetch("/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ [name]: value }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || "Save failed");
    const data = await res.json();
    showStatus(`${name} updated`, true);
    render(data);
  } catch (e) {
    showStatus(e.message, false);
    loadSettings();
  }
}

async function resetOne(name) {
  const res = await fetch("/settings/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  const data = await res.json();
  showStatus(`${name} reset to default`, true);
  render(data);
}

resetAllBtn.onclick = async () => {
  if (!confirm("Reset ALL settings to config.py defaults?")) return;
  const res = await fetch("/settings/reset-all", { method: "POST" });
  const data = await res.json();
  showStatus("All settings reset to defaults", true);
  render(data);
};

loadSettings();
