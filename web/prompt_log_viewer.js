const ROLE_COLORS = {
  system: 'role-system', user: 'role-user', assistant: 'role-assistant',
  tool: 'role-tool', tools: 'role-tools', console: 'role-console',
  response: 'role-response', thinking: 'role-thinking',
};

let allEntries = [];
let groups = [];
let activeRoles = new Set();
let activeSections = new Set();
let selectedEntryIndex = null; // null = show all requests

// Annotations for the currently loaded file. Notes are keyed by position,
// not by iteration number - see the comment above _annotations_path() in
// main.py for why. iteration_notes["3"] is the note on groups[3] as a
// whole; chunk_notes["3:1"] is the note on groupChunks(groups[3])[1].
let annotations = { iteration_notes: {}, chunk_notes: {} };
let currentFilename = null;
let saveTimer = null;

// Older log files (written before console capture existed) have entries
// with no "type" field at all - infer "prompt" for those from the
// presence of "chunks", "console" otherwise, so old logs still load fine.
function entryType(e) {
  return e.type || (e.chunks ? 'prompt' : 'console');
}

// Console entries are always written immediately after the prompt entry
// they belong to (see main.py's _log_console). Pair them up here so the
// rest of the page can treat "one iteration" as one thing to render,
// instead of two separate array elements. A console entry with no
// preceding prompt (rare - e.g. the max-iterations bail-out case) still
// gets its own standalone group so nothing is silently dropped.
function buildGroups(entries) {
  const out = [];
  let i = 0;
  while (i < entries.length) {
    const e = entries[i];
    if (entryType(e) === 'console') {
      out.push({ prompt: null, console: e });
      i++;
      continue;
    }
    const group = { prompt: e, console: null };
    const next = entries[i + 1];
    if (next && entryType(next) === 'console') {
      group.console = next;
      i += 2;
    } else {
      i += 1;
    }
    out.push(group);
  }
  return out;
}

// A group's console lines and the model's actual output for that
// iteration are exposed as ordinary chunks, so the existing role/section
// filter + collapsible-chunk UI handles both for free.
function groupChunks(group) {
  const chunks = group.prompt ? [...group.prompt.chunks] : [];
  if (group.console) {
    if (group.console.lines && group.console.lines.length) {
      chunks.push({
        role: 'console',
        section: 'console_output',
        content: group.console.lines.join('\n'),
      });
    }
    if (group.console.thinking) {
      chunks.push({
        role: 'thinking',
        section: 'reasoning',
        content: group.console.thinking,
      });
    }
    if (group.console.response) {
      chunks.push({
        role: 'response',
        section: group.console.response.kind || 'response',
        content: group.console.response.text || '',
      });
    }
  }
  return chunks;
}

function groupRef(group) {
  return group.prompt || group.console;
}

// Pulls select_tools()'s tier label ("direct"/"context_widened"/"rescue"/
// "core_only"/"disabled"/"empty" - see main.py) straight off the raw
// tool_selection_debug chunk written by prompt_log_engine.py's
// log_prompt(). Deliberately reads group.prompt.chunks directly rather
// than going through groupChunks()'s role/section-filtered view, so the
// iteration-head badge stays visible even if the "tools" role or that
// section happens to be filtered out right now. Older log entries (from
// before this field existed) have no "tier" key on the chunk - returns
// null for those, same as if tool selection had never run.
function groupToolTier(group) {
  const chunks = group.prompt?.chunks;
  if (!chunks) return null;
  const debugChunk = chunks.find(c => c.section === 'tool_selection_debug');
  return debugChunk?.tier || null;
}

function tierBadge(tier) {
  if (!tier) return '';
  return `<span class="badge tier-${tier}" title="tool selection tier">${tier.replace(/_/g, ' ')}</span>`;
}

// ---------------------------------------------------------------------
// Annotations: load/save against the backend, plus the small inline
// widget (pinned note / "+ note" button / textarea) shared by both
// iteration-level and chunk-level notes.
// ---------------------------------------------------------------------

async function loadAnnotationsForFile(filename) {
  currentFilename = filename;
  try {
    const res = await fetch(`/prompt-logs/${encodeURIComponent(filename)}/annotations`);
    annotations = await res.json();
  } catch (err) {
    console.error('Failed to load annotations:', err);
    annotations = { iteration_notes: {}, chunk_notes: {} };
  }
  if (!annotations.iteration_notes) annotations.iteration_notes = {};
  if (!annotations.chunk_notes) annotations.chunk_notes = {};
  setSaveStatus('');
}

function scheduleSaveAnnotations() {
  setSaveStatus('unsaved');
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveAnnotationsNow, 800);
}

async function saveAnnotationsNow() {
  if (!currentFilename) return;
  setSaveStatus('saving');
  try {
    const res = await fetch(`/prompt-logs/${encodeURIComponent(currentFilename)}/annotations`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(annotations),
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    setSaveStatus('saved');
  } catch (err) {
    console.error('Failed to save annotations:', err);
    setSaveStatus('error');
  }
}

function setSaveStatus(state) {
  const el = document.getElementById('saveStatus');
  if (!el) return;
  el.classList.remove('saving', 'error');
  if (state === 'unsaved') el.textContent = 'Unsaved changes';
  else if (state === 'saving') { el.textContent = 'Saving\u2026'; el.classList.add('saving'); }
  else if (state === 'saved') el.textContent = 'Notes saved';
  else if (state === 'error') { el.textContent = 'Save failed'; el.classList.add('error'); }
  else el.textContent = '';
}

// notesStore is either annotations.iteration_notes or annotations.chunk_notes;
// key is that entry's position-based key within it (see comment above the
// `annotations` declaration).
function createNoteWidget(notesStore, key, placeholder) {
  const wrap = document.createElement('div');
  wrap.className = 'note-wrap';

  function renderView() {
    wrap.innerHTML = '';
    const text = notesStore[key];
    if (text) {
      const pinned = document.createElement('div');
      pinned.className = 'note-pinned';
      pinned.textContent = text;
      pinned.title = 'Click to edit note';
      pinned.addEventListener('click', (e) => { e.stopPropagation(); renderEdit(); });
      wrap.appendChild(pinned);
    } else {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'note-btn';
      btn.textContent = '+ note';
      btn.addEventListener('click', (e) => { e.stopPropagation(); renderEdit(); });
      wrap.appendChild(btn);
    }
  }

  function renderEdit() {
    wrap.innerHTML = '';
    const ta = document.createElement('textarea');
    ta.className = 'note-textarea';
    ta.value = notesStore[key] || '';
    ta.placeholder = placeholder;
    ta.addEventListener('click', (e) => e.stopPropagation());
    ta.addEventListener('keydown', (e) => e.stopPropagation());
    ta.addEventListener('input', () => {
      if (ta.value.trim()) notesStore[key] = ta.value;
      else delete notesStore[key];
      scheduleSaveAnnotations();
    });
    ta.addEventListener('blur', () => {
      renderView();
      buildRequestList();
    });
    wrap.appendChild(ta);
    ta.focus();
  }

  renderView();
  return wrap;
}

async function loadFileList() {
  const res = await fetch('/prompt-logs');
  const files = await res.json();
  const sel = document.getElementById('fileSelect');
  sel.innerHTML = '';
  if (!files.length) {
    document.getElementById('status').textContent = 'No log files found in prompt_logs/ yet - send a message through SillyTavern first.';
    return;
  }
  for (const f of files) {
    const opt = document.createElement('option');
    opt.value = f.filename;
    opt.textContent = `${f.filename}  (${f.size_kb} KB)`;
    sel.appendChild(opt);
  }
  sel.addEventListener('change', () => loadFile(sel.value));
  await loadFile(files[0].filename);
}

async function loadFile(filename) {
  document.getElementById('status').textContent = 'Loading ' + filename + '...';
  const res = await fetch('/prompt-logs/' + encodeURIComponent(filename));
  allEntries = await res.json();
  groups = buildGroups(allEntries);
  selectedEntryIndex = null;
  await loadAnnotationsForFile(filename);
  buildRequestList();
  buildFilterLists();
  render();
}

function buildRequestList() {
  const box = document.getElementById('requestFilter');
  box.innerHTML = '';

  const allItem = document.createElement('div');
  allItem.className = 'req-item' + (selectedEntryIndex === null ? ' active' : '');
  allItem.innerHTML = `<span class="req-title">All requests</span><span class="req-meta">${groups.length} total</span>`;
  allItem.addEventListener('click', () => {
    selectedEntryIndex = null;
    buildRequestList();
    render();
  });
  box.appendChild(allItem);

  groups.forEach((group, idx) => {
    const ref = groupRef(group);
    const time = (ref.timestamp || '').split('T')[1]?.split('.')[0] || ref.timestamp || '?';
    const consoleMark = group.console ? ' \u{1F5A5}' : ''; // small monitor icon = has console output
    const note = annotations.iteration_notes[String(idx)];
    const item = document.createElement('div');
    item.className = 'req-item' + (selectedEntryIndex === idx ? ' active' : '');
    item.innerHTML = `
      <span class="req-title">iteration ${ref.iteration} &middot; ${time}${consoleMark}${note ? ' <span class="req-note-marker">\u{1F4DD}</span>' : ''}</span>
      <span class="req-meta">${groupChunks(group).length} chunks</span>`;
    if (note) item.querySelector('.req-note-marker').title = note;
    item.addEventListener('click', () => {
      selectedEntryIndex = idx;
      buildRequestList();
      render();
    });
    box.appendChild(item);
  });
}

function buildFilterLists() {
  const roles = new Set();
  const sections = new Set();
  for (const group of groups) {
    for (const c of groupChunks(group)) {
      roles.add(c.role);
      sections.add(c.section);
    }
  }
  activeRoles = new Set(roles);
  activeSections = new Set(sections);

  const roleBox = document.getElementById('roleFilters');
  roleBox.innerHTML = '';
  for (const r of [...roles].sort()) {
    const id = 'role-' + r;
    const colorKey = ROLE_COLORS[r] ? ROLE_COLORS[r].replace('role-', '') : 'tools';
    roleBox.insertAdjacentHTML('beforeend', `
      <label class="chk">
        <input type="checkbox" id="${id}" checked>
        <span class="dot" style="background:var(--c-${colorKey})"></span>
        ${r}
      </label>`);
  }
  roleBox.querySelectorAll('input').forEach(inp => {
    inp.addEventListener('change', () => {
      const r = inp.id.replace('role-', '');
      inp.checked ? activeRoles.add(r) : activeRoles.delete(r);
      render();
    });
  });

  const secBox = document.getElementById('sectionFilters');
  secBox.innerHTML = '';
  for (const s of [...sections].sort()) {
    const id = 'sec-' + s;
    secBox.insertAdjacentHTML('beforeend', `
      <label class="chk">
        <input type="checkbox" id="${id}" checked>
        ${s}
      </label>`);
  }
  secBox.querySelectorAll('input').forEach(inp => {
    inp.addEventListener('change', () => {
      const s = inp.id.replace('sec-', '');
      inp.checked ? activeSections.add(s) : activeSections.delete(s);
      render();
    });
  });
}

function escapeHtml(str) {
  return str.replace(/[&<>"']/g, ch => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[ch]));
}

function highlight(text, query) {
  const escaped = escapeHtml(text);
  if (!query) return escaped;
  const safeQuery = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return escaped.replace(new RegExp('(' + safeQuery + ')', 'ig'), '<mark>$1</mark>');
}

function render() {
  const query = document.getElementById('searchBox').value.trim();
  const resultsEl = document.getElementById('results');
  resultsEl.innerHTML = '';
  let shownIterations = 0;
  let shownChunks = 0;

  const groupsToShow = selectedEntryIndex === null
    ? groups.map((g, i) => [i, g])
    : [[selectedEntryIndex, groups[selectedEntryIndex]]];

  for (const [groupIndex, group] of groupsToShow) {
    if (!group) continue;
    const ref = groupRef(group);
    const allChunks = groupChunks(group);
    const matchingChunks = [];
    allChunks.forEach((c, chunkIndex) => {
      if (!activeRoles.has(c.role)) return;
      if (!activeSections.has(c.section)) return;
      if (query && !c.content.toLowerCase().includes(query.toLowerCase())) return;
      matchingChunks.push({ ...c, _chunkIndex: chunkIndex });
    });
    if (!matchingChunks.length) continue;
    shownIterations++;

    const iterDiv = document.createElement('div');
    iterDiv.className = 'iteration';
    iterDiv.dataset.groupIndex = String(groupIndex);
    iterDiv.innerHTML = `
      <div class="iteration-head">
        <span>${ref.timestamp}</span>
        <span>iteration ${ref.iteration} &middot; model: ${group.prompt?.model || '?'}</span>
        ${tierBadge(groupToolTier(group))}
      </div>`;
    iterDiv.querySelector('.iteration-head').appendChild(
      createNoteWidget(annotations.iteration_notes, String(groupIndex), 'Note about this iteration\u2026')
    );

    for (const c of matchingChunks) {
      shownChunks++;
      const roleClass = ROLE_COLORS[c.role] || 'role-tools';
      const chunkDiv = document.createElement('div');
      chunkDiv.className = 'chunk'
        + (c.role === 'console' ? ' role-console-chunk' : '')
        + (c.role === 'response' ? ' role-response-chunk' : '')
        + (c.role === 'thinking' ? ' role-thinking-chunk' : '');
      chunkDiv.dataset.chunkIndex = String(c._chunkIndex);
      chunkDiv.innerHTML = `
        <div class="chunk-head">
          <span class="badge ${roleClass}">${c.role}</span>
          <span class="section-name">${c.section}</span>
          ${tierBadge(c.tier)}
          <span class="char-count">${c.content.length.toLocaleString()} chars</span>
        </div>
        <div class="chunk-body"><pre>${highlight(c.content, query)}</pre></div>`;
      const head = chunkDiv.querySelector('.chunk-head');
      const body = chunkDiv.querySelector('.chunk-body');
      head.appendChild(
        createNoteWidget(annotations.chunk_notes, `${groupIndex}:${c._chunkIndex}`, 'Note about this chunk\u2026')
      );
      head.addEventListener('click', () => body.classList.toggle('open'));
      if (query) body.classList.add('open');
      iterDiv.appendChild(chunkDiv);
    }
    resultsEl.appendChild(iterDiv);
  }

  const focusNote = selectedEntryIndex === null
    ? (groups.length ? ` (of ${groups.length} logged requests total)` : '')
    : ` (focused on iteration ${groupRef(groups[selectedEntryIndex])?.iteration})`;
  document.getElementById('status').textContent =
    `${shownIterations} request${shownIterations === 1 ? '' : 's'} / ${shownChunks} chunk${shownChunks === 1 ? '' : 's'} shown`
    + focusNote;

  if (!shownChunks) {
    resultsEl.innerHTML = '<div class="empty">No chunks match the current filters.</div>';
  }
}

// ---------------------------------------------------------------------
// Cross-session note search. The heavy lifting (reading every file's
// annotations, resolving matches back to their iteration/chunk) happens
// server-side in prompt_log_engine.py's /notes/search - this is just a
// fetch and a shape adjustment for the render/navigate functions below.
// ---------------------------------------------------------------------

let noteSearchDebounce = null;
let noteSearchRequestId = 0; // guards against an older, slower request's response landing after a newer one

async function runNoteSearch(query) {
  const res = await fetch('/notes/search?q=' + encodeURIComponent(query));
  if (!res.ok) throw new Error('HTTP ' + res.status);
  const data = await res.json();
  return (data.results || []).map(r => ({
    filename: r.filename,
    groupIndex: r.group_index,
    chunkIndex: r.chunk_index,
    kind: r.kind,
    chunk: r.chunk,
    noteText: r.note_text,
    ref: { timestamp: r.timestamp, iteration: r.iteration, model: r.model },
  }));
}

function renderNoteSearchResults(results, query) {
  const box = document.getElementById('noteSearchResults');
  const status = document.getElementById('noteSearchStatus');
  box.innerHTML = '';

  if (!query) {
    status.textContent = 'Type to search note content across every session.';
    return;
  }
  status.textContent = `${results.length} matching note${results.length === 1 ? '' : 's'}`;
  if (!results.length) return;

  for (const r of results) {
    const time = (r.ref.timestamp || '').split('T')[1]?.split('.')[0] || r.ref.timestamp || '?';
    const card = document.createElement('div');
    card.className = 'note-search-result';

    const meta = document.createElement('div');
    meta.className = 'note-search-meta';
    if (r.kind === 'chunk') {
      const roleClass = ROLE_COLORS[r.chunk.role] || 'role-tools';
      meta.innerHTML = `
        <span>${escapeHtml(r.filename)}</span>
        <span>&middot;</span>
        <span>iteration ${r.ref.iteration} &middot; ${time}</span>
        <span>&middot;</span>
        <span class="badge ${roleClass}">${r.chunk.role}</span>
        <span>${escapeHtml(r.chunk.section)}</span>`;
    } else {
      meta.innerHTML = `
        <span>${escapeHtml(r.filename)}</span>
        <span>&middot;</span>
        <span>iteration ${r.ref.iteration} &middot; ${time}</span>
        <span>&middot;</span>
        <span>iteration-level note</span>`;
    }

    const note = document.createElement('div');
    note.className = 'note-search-note';
    note.innerHTML = highlight(r.noteText, query);

    card.appendChild(meta);
    card.appendChild(note);

    if (r.kind === 'chunk' && r.chunk) {
      const pre = document.createElement('pre');
      pre.className = 'note-search-chunk-content';
      pre.textContent = r.chunk.content;
      card.appendChild(pre);
    } else {
      const info = document.createElement('div');
      info.className = 'note-search-chunk-content';
      info.textContent = `Note on the whole iteration (model: ${r.ref.model || '?'}).`;
      card.appendChild(info);
    }

    const open = document.createElement('div');
    open.className = 'note-search-open';
    open.textContent = 'Open in viewer \u2192';
    card.appendChild(open);

    card.addEventListener('click', () => openNoteSearchResult(r));
    box.appendChild(card);
  }
}

async function openNoteSearchResult(result) {
  closeNoteSearch();
  const sel = document.getElementById('fileSelect');
  if (sel.value !== result.filename) {
    sel.value = result.filename;
    await loadFile(result.filename);
  }
  document.getElementById('searchBox').value = '';
  selectedEntryIndex = result.groupIndex;

  if (result.kind === 'chunk' && result.chunk) {
    activeRoles.add(result.chunk.role);
    activeSections.add(result.chunk.section);
    const roleCb = document.getElementById('role-' + result.chunk.role);
    if (roleCb) roleCb.checked = true;
    const secCb = document.getElementById('sec-' + result.chunk.section);
    if (secCb) secCb.checked = true;
  }

  buildRequestList();
  render();

  requestAnimationFrame(() => {
    const iterDiv = document.querySelector(`.iteration[data-group-index="${result.groupIndex}"]`);
    if (!iterDiv) return;
    const chunkDiv = result.kind === 'chunk'
      ? iterDiv.querySelector(`.chunk[data-chunk-index="${result.chunkIndex}"]`)
      : null;
    const target = chunkDiv || iterDiv;
    target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    if (chunkDiv) chunkDiv.querySelector('.chunk-body')?.classList.add('open');
    target.classList.add('flash-target');
    setTimeout(() => target.classList.remove('flash-target'), 1800);
  });
}

function openNoteSearch() {
  document.getElementById('noteSearchOverlay').classList.remove('hidden');
  const input = document.getElementById('noteSearchInput');
  input.value = '';
  document.getElementById('noteSearchResults').innerHTML = '';
  document.getElementById('noteSearchStatus').textContent = 'Type to search note content across every session.';
  input.focus();
}

function closeNoteSearch() {
  document.getElementById('noteSearchOverlay').classList.add('hidden');
}

document.getElementById('noteSearchBtn').addEventListener('click', openNoteSearch);
document.getElementById('noteSearchCloseBtn').addEventListener('click', closeNoteSearch);
document.getElementById('noteSearchOverlay').addEventListener('click', (e) => {
  if (e.target.id === 'noteSearchOverlay') closeNoteSearch();
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !document.getElementById('noteSearchOverlay').classList.contains('hidden')) {
    closeNoteSearch();
  }
});
document.getElementById('noteSearchInput').addEventListener('input', (e) => {
  const query = e.target.value.trim();
  clearTimeout(noteSearchDebounce);
  if (!query) {
    renderNoteSearchResults([], '');
    return;
  }
  document.getElementById('noteSearchStatus').textContent = 'Searching\u2026';
  const requestId = ++noteSearchRequestId;
  noteSearchDebounce = setTimeout(async () => {
    try {
      const results = await runNoteSearch(query);
      if (requestId !== noteSearchRequestId) return; // a newer keystroke already superseded this request
      renderNoteSearchResults(results, query);
    } catch (err) {
      console.error('Note search failed:', err);
      if (requestId !== noteSearchRequestId) return;
      document.getElementById('noteSearchStatus').textContent = 'Search failed - is the server running?';
      document.getElementById('noteSearchResults').innerHTML = '';
    }
  }, 300);
});

document.getElementById('searchBox').addEventListener('input', render);
document.getElementById('expandAllBtn').addEventListener('click', () => {
  document.querySelectorAll('.chunk-body').forEach(b => b.classList.add('open'));
});
document.getElementById('collapseAllBtn').addEventListener('click', () => {
  document.querySelectorAll('.chunk-body').forEach(b => b.classList.remove('open'));
});

loadFileList();
