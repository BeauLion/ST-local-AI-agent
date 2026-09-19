// Floating chat widget - talks to this server's own /webchat/completions
// (same agent loop as SillyTavern's /v1/chat/completions, just without the
// Bearer-key gate meant for that client - see main.py) and to /personas for
// the Personas tab. Self-contained: builds its own DOM. Chat history is kept
// in memory only (cleared on reload); personas are backed by personas.json
// on the server, so they persist and are what /webchat/completions reads to
// decide which system message to prepend.
(function () {
  const messages = []; // {role, content}
  let personas = [];
  let selectedPersonaId = null;
  let editingPersonaId = null; // id being edited inline, or "new"

  const fab = document.createElement("button");
  fab.className = "cw-fab";
  fab.type = "button";
  fab.title = "Chat with the agent";
  fab.innerHTML = "&#128172;";

  const panel = document.createElement("div");
  panel.className = "cw-panel";
  panel.hidden = true;
  panel.innerHTML = `
    <div class="cw-header">
      <span class="cw-title">Agent Chat</span>
      <button type="button" class="cw-close" title="Close">&times;</button>
    </div>
    <div class="cw-tabs">
      <button type="button" class="cw-tab active" data-tab="chat">Chat</button>
      <button type="button" class="cw-tab" data-tab="personas">Personas</button>
    </div>
    <div class="cw-view cw-view-chat">
      <div class="cw-messages"></div>
      <div class="cw-inputRow">
        <textarea placeholder="Ask something..." rows="1"></textarea>
        <button type="button" class="cw-send">Send</button>
      </div>
    </div>
    <div class="cw-view cw-view-personas" hidden>
      <div class="cw-personaList"></div>
      <div class="cw-personaFooter">
        <button type="button" class="cw-personaAddBtn">+ New persona</button>
      </div>
    </div>
  `;

  document.body.appendChild(fab);
  document.body.appendChild(panel);

  const messagesEl = panel.querySelector(".cw-messages");
  const textarea = panel.querySelector("textarea");
  const sendBtn = panel.querySelector(".cw-send");
  const closeBtn = panel.querySelector(".cw-close");
  const tabButtons = panel.querySelectorAll(".cw-tab");
  const viewChat = panel.querySelector(".cw-view-chat");
  const viewPersonas = panel.querySelector(".cw-view-personas");
  const personaListEl = panel.querySelector(".cw-personaList");
  const personaAddBtn = panel.querySelector(".cw-personaAddBtn");

  function addBubble(role, text) {
    const el = document.createElement("div");
    el.className = "cw-msg " + role;
    el.textContent = text;
    messagesEl.appendChild(el);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return el;
  }

  function setOpen(open) {
    panel.hidden = !open;
    if (open) {
      textarea.focus();
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
  }

  fab.onclick = () => setOpen(panel.hidden);
  closeBtn.onclick = () => setOpen(false);

  tabButtons.forEach((btn) => {
    btn.onclick = () => {
      tabButtons.forEach((b) => b.classList.toggle("active", b === btn));
      const tab = btn.dataset.tab;
      viewChat.hidden = tab !== "chat";
      viewPersonas.hidden = tab !== "personas";
      if (tab === "personas") loadPersonas();
    };
  });

  async function send() {
    const text = textarea.value.trim();
    if (!text) return;
    textarea.value = "";
    textarea.style.height = "36px";
    messages.push({ role: "user", content: text });
    addBubble("user", text);

    sendBtn.disabled = true;
    const pending = addBubble("assistant pending", "Thinking...");

    try {
      const resp = await fetch("/webchat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: "agent", stream: false, messages }),
      });
      const data = await resp.json();
      pending.remove();
      if (!resp.ok) {
        const detail = (data && data.error && data.error.message) || resp.statusText;
        addBubble("error", detail);
        return;
      }
      const reply = data.choices && data.choices[0] && data.choices[0].message;
      const content = (reply && reply.content) || "(no response)";
      messages.push({ role: "assistant", content });
      addBubble("assistant", content);
    } catch (err) {
      pending.remove();
      addBubble("error", "Request failed: " + err.message);
    } finally {
      sendBtn.disabled = false;
      textarea.focus();
    }
  }

  sendBtn.onclick = send;
  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
  textarea.addEventListener("input", () => {
    textarea.style.height = "36px";
    textarea.style.height = Math.min(textarea.scrollHeight, 100) + "px";
  });

  // ---------------------------- Personas tab ----------------------------

  async function loadPersonas() {
    try {
      const resp = await fetch("/personas");
      const data = await resp.json();
      personas = data.personas || [];
      selectedPersonaId = data.selected_persona_id || null;
      renderPersonas();
    } catch (err) {
      personaListEl.textContent = "Failed to load personas: " + err.message;
    }
  }

  function renderPersonas() {
    personaListEl.innerHTML = "";

    personas.forEach((p) => {
      if (editingPersonaId === p.id) {
        personaListEl.appendChild(buildPersonaEditForm(p));
        return;
      }
      const row = document.createElement("div");
      row.className = "cw-personaRow" + (p.id === selectedPersonaId ? " selected" : "");

      const nameBtn = document.createElement("button");
      nameBtn.type = "button";
      nameBtn.className = "cw-personaName";
      nameBtn.textContent = p.name;
      nameBtn.title = "Select this persona";
      nameBtn.onclick = () => selectPersona(p.id === selectedPersonaId ? null : p.id);
      row.appendChild(nameBtn);

      const actions = document.createElement("div");
      actions.className = "cw-personaActions";

      const editBtn = document.createElement("button");
      editBtn.type = "button";
      editBtn.title = "Edit";
      editBtn.textContent = "✎";
      editBtn.onclick = () => {
        editingPersonaId = p.id;
        renderPersonas();
      };
      actions.appendChild(editBtn);

      const delBtn = document.createElement("button");
      delBtn.type = "button";
      delBtn.title = "Delete";
      delBtn.textContent = "✖";
      delBtn.onclick = () => deletePersona(p.id);
      actions.appendChild(delBtn);

      row.appendChild(actions);
      personaListEl.appendChild(row);
    });

    if (editingPersonaId === "new") {
      personaListEl.appendChild(buildPersonaEditForm(null));
    }

    if (!personas.length && editingPersonaId !== "new") {
      const empty = document.createElement("p");
      empty.className = "cw-personaEmpty";
      empty.textContent = "No personas yet.";
      personaListEl.appendChild(empty);
    }
  }

  function buildPersonaEditForm(persona) {
    const wrap = document.createElement("div");
    wrap.className = "cw-personaForm";

    const nameInput = document.createElement("input");
    nameInput.type = "text";
    nameInput.placeholder = "Persona name";
    nameInput.value = persona ? persona.name : "";
    wrap.appendChild(nameInput);

    const contentInput = document.createElement("textarea");
    contentInput.placeholder = "System prompt content for this persona...";
    contentInput.value = persona ? persona.content : "";
    wrap.appendChild(contentInput);

    const actions = document.createElement("div");
    actions.className = "cw-personaFormActions";

    const cancelBtn = document.createElement("button");
    cancelBtn.type = "button";
    cancelBtn.textContent = "Cancel";
    cancelBtn.onclick = () => {
      editingPersonaId = null;
      renderPersonas();
    };
    actions.appendChild(cancelBtn);

    const saveBtn = document.createElement("button");
    saveBtn.type = "button";
    saveBtn.className = "primary";
    saveBtn.textContent = "Save";
    saveBtn.onclick = () => savePersona(persona ? persona.id : null, nameInput.value, contentInput.value);
    actions.appendChild(saveBtn);

    wrap.appendChild(actions);
    return wrap;
  }

  async function savePersona(personaId, name, content) {
    name = name.trim();
    if (!name) {
      alert("Persona name cannot be empty.");
      return;
    }
    try {
      const resp = await fetch(personaId ? `/personas/${personaId}` : "/personas", {
        method: personaId ? "PATCH" : "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, content }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        alert((data && data.detail) || "Failed to save persona.");
        return;
      }
      editingPersonaId = null;
      await loadPersonas();
    } catch (err) {
      alert("Failed to save persona: " + err.message);
    }
  }

  async function deletePersona(personaId) {
    if (!confirm("Delete this persona?")) return;
    try {
      const resp = await fetch(`/personas/${personaId}`, { method: "DELETE" });
      if (!resp.ok) {
        const data = await resp.json();
        alert((data && data.detail) || "Failed to delete persona.");
        return;
      }
      await loadPersonas();
    } catch (err) {
      alert("Failed to delete persona: " + err.message);
    }
  }

  async function selectPersona(personaId) {
    try {
      const resp = await fetch("/personas/select", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ persona_id: personaId }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        alert((data && data.detail) || "Failed to select persona.");
        return;
      }
      personas = data.personas || [];
      selectedPersonaId = data.selected_persona_id || null;
      renderPersonas();
    } catch (err) {
      alert("Failed to select persona: " + err.message);
    }
  }

  personaAddBtn.onclick = () => {
    editingPersonaId = "new";
    renderPersonas();
  };
})();
