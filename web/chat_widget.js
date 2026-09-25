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
    if (role === "assistant") {
      el.classList.add("cw-md");
      el.innerHTML = renderMarkdown(text);
    } else {
      el.textContent = text;
    }
    messagesEl.appendChild(el);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return el;
  }

  // ------------------------- Minimal markdown renderer -------------------------
  // Small, dependency-free subset (headers, bold/italic, inline/fenced code,
  // links, lists, blockquotes, paragraphs) - enough for typical model output
  // without pulling in a vendored library. Input is HTML-escaped up front so
  // nothing the model writes (including raw HTML) is interpreted as markup.

  function escapeHtml(s) {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function renderInline(text) {
    text = text.replace(/`([^`]+)`/g, (m, c) => `<code>${c}</code>`);
    text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    text = text.replace(/__([^_]+)__/g, "<strong>$1</strong>");
    text = text.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    text = text.replace(/(^|[^\w])_([^_]+)_(?!\w)/g, "$1<em>$2</em>");
    text = text.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    return text;
  }

  function renderMarkdown(raw) {
    const lines = escapeHtml(raw).split(/\r?\n/);
    let html = "";
    let i = 0;
    let listBuffer = null; // { type: "ul"|"ol", items: [] }

    function flushList() {
      if (!listBuffer) return;
      const tag = listBuffer.type;
      html += `<${tag}>${listBuffer.items.map((it) => `<li>${renderInline(it)}</li>`).join("")}</${tag}>`;
      listBuffer = null;
    }

    while (i < lines.length) {
      const line = lines[i];

      const fence = line.match(/^```(\w*)\s*$/);
      if (fence) {
        flushList();
        const codeLines = [];
        i++;
        while (i < lines.length && !/^```\s*$/.test(lines[i])) {
          codeLines.push(lines[i]);
          i++;
        }
        i++; // skip closing fence (if any)
        html += `<pre><code>${codeLines.join("\n")}</code></pre>`;
        continue;
      }

      if (!line.trim()) {
        flushList();
        i++;
        continue;
      }

      const header = line.match(/^(#{1,6})\s+(.*)$/);
      if (header) {
        flushList();
        const level = header[1].length;
        html += `<h${level}>${renderInline(header[2])}</h${level}>`;
        i++;
        continue;
      }

      if (/^&gt;\s?/.test(line)) {
        flushList();
        const quoteLines = [];
        while (i < lines.length && /^&gt;\s?/.test(lines[i])) {
          quoteLines.push(lines[i].replace(/^&gt;\s?/, ""));
          i++;
        }
        html += `<blockquote>${renderInline(quoteLines.join(" "))}</blockquote>`;
        continue;
      }

      const ul = line.match(/^[-*]\s+(.*)$/);
      if (ul) {
        if (!listBuffer || listBuffer.type !== "ul") { flushList(); listBuffer = { type: "ul", items: [] }; }
        listBuffer.items.push(ul[1]);
        i++;
        continue;
      }

      const ol = line.match(/^\d+\.\s+(.*)$/);
      if (ol) {
        if (!listBuffer || listBuffer.type !== "ol") { flushList(); listBuffer = { type: "ol", items: [] }; }
        listBuffer.items.push(ol[1]);
        i++;
        continue;
      }

      flushList();
      const paraLines = [line];
      i++;
      while (
        i < lines.length && lines[i].trim() &&
        !/^```/.test(lines[i]) && !/^#{1,6}\s/.test(lines[i]) &&
        !/^[-*]\s+/.test(lines[i]) && !/^\d+\.\s+/.test(lines[i]) && !/^&gt;/.test(lines[i])
      ) {
        paraLines.push(lines[i]);
        i++;
      }
      html += `<p>${renderInline(paraLines.join("<br>"))}</p>`;
    }
    flushList();
    return html;
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

  // Public API for other widgets on the page (e.g. gantt.js's task planner)
  // to hand off a prompt: opens the panel on the chat tab and fills the
  // input, leaving it to the user to review and hit send.
  function pasteToInput(text) {
    setOpen(true);
    tabButtons.forEach((b) => b.classList.toggle("active", b.dataset.tab === "chat"));
    viewChat.hidden = false;
    viewPersonas.hidden = true;
    textarea.value = text;
    textarea.style.height = "36px";
    textarea.style.height = Math.min(textarea.scrollHeight, 100) + "px";
    textarea.focus();
  }

  window.chatWidget = { pasteToInput };
})();
