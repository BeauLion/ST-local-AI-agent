(function () {
  const statusBox = document.getElementById("statusBox");
  const projectSelect = document.getElementById("projectSelect");
  const startInput = document.getElementById("startInput");
  const endInput = document.getElementById("endInput");
  const ganttEmpty = document.getElementById("ganttEmpty");
  const chartArea = document.getElementById("chartArea");
  const taskSidebar = document.getElementById("taskSidebar");
  const taskSidebarContent = document.getElementById("taskSidebarContent");
  const sidebarToggle = document.getElementById("sidebarToggle");
  const zoomOutBtn = document.getElementById("zoomOutBtn");
  const zoomInBtn = document.getElementById("zoomInBtn");
  const datesToggle = document.getElementById("datesToggle");
  const datesBody = document.getElementById("datesBody");
  const board = document.getElementById("board");
  const boardHint = document.getElementById("boardHint");
  const boardEditToggle = document.getElementById("boardEditToggle");
  const boardDeleteToggle = document.getElementById("boardDeleteToggle");
  const chartEditToggle = document.getElementById("chartEditToggle");
  let ganttInstance = null;
  let projectsById = {};
  let currentViewMode = "Week";
  let activeDrag = null; // { taskId, side } while a bar's left/right handle is being dragged
  let dragLabelEl = null;
  let boardProjectId = null;
  let boardEditMode = false;
  let boardDeleteMode = false;
  let chartEditMode = false;
  const PROJECT_STATUSES = [
    { key: "active", label: "Active" },
    { key: "paused", label: "Paused" },
    { key: "completed", label: "Completed" },
  ];
  const TASK_STATUSES = [
    { key: "pending", label: "Pending" },
    { key: "active", label: "Active" },
    { key: "blocked", label: "Blocked" },
    { key: "done", label: "Done" },
    { key: "cancelled", label: "Cancelled" },
  ];
  const EFFORT_OPTIONS = [["", "Effort (unset)"], ["low", "Low effort"], ["medium", "Medium effort"], ["high", "High effort"]];
  const WHEN_TIME_OPTIONS = [["", "When (unset)"], ["morning", "Morning"], ["afternoon", "Afternoon"], ["evening", "Evening"]];
  const WHEN_MODIFIER_OPTIONS = [["", "Any day"], ["weekday", "Weekday"], ["weekend", "Weekend"]];

  // ---------------------------------------------------------------------
  // Small shared helpers - option-list building and the fetch/JSON/error
  // pattern repeated across every read and every save action below.
  // ---------------------------------------------------------------------

  // Fills a <select> with options built from either an array of
  // {key,label} objects (project/task statuses) or an array of
  // [value, label] tuples (the small inline effort/when selects).
  function fillOptions(selectEl, items, opts) {
    const getValue = opts && opts.value ? opts.value : (item) => item[0];
    const getLabel = opts && opts.label ? opts.label : (item) => item[1];
    selectEl.innerHTML = "";
    for (const item of items) {
      const option = document.createElement("option");
      option.value = getValue(item);
      option.textContent = getLabel(item);
      selectEl.appendChild(option);
    }
  }

  function fillStatusOptions(selectEl, statuses) {
    fillOptions(selectEl, statuses, { value: (s) => s.key, label: (s) => s.label });
  }

  // Wraps fetch() with the JSON-body/error-detail handling every write
  // (and most reads) below need: throws Error(<server detail or
  // fallbackMessage>) on a non-2xx response, otherwise resolves to the
  // parsed JSON body (or null for a 204).
  async function apiFetch(url, options, fallbackMessage) {
    const res = await fetch(url, options);
    if (!res.ok) {
      let detail = fallbackMessage || "Request failed";
      try {
        const data = await res.json();
        if (data && data.detail) detail = data.detail;
      } catch (e) { /* body wasn't JSON - keep fallbackMessage */ }
      throw new Error(detail);
    }
    if (res.status === 204) return null;
    return res.json();
  }

  function jsonRequest(method, body) {
    return { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  }

  function setViewMode(mode) {
    currentViewMode = mode;
    if (ganttInstance) ganttInstance.change_view_mode(mode);
    zoomOutBtn.classList.toggle("active", mode === "Month");
    zoomInBtn.classList.toggle("active", mode === "Week");
  }

  function showStatus(message, ok) {
    statusBox.textContent = message;
    statusBox.className = "status " + (ok ? "ok" : "err");
    statusBox.hidden = false;
    setTimeout(() => { statusBox.hidden = true; }, 4000);
  }

  function formatDate(iso) {
    if (!iso) return "";
    // completed_at is a full UTC ISO timestamp; show just the date part
    // in the browser's local time so it reads naturally next to the task.
    const d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
  }

  function formatDeadline(value) {
    if (!value) return "";
    // "YYYY-MM-DD" (date only) or "YYYY-MM-DDTHH:MM" (date+time), matching
    // the <input type="date"|"datetime-local"> value format - see
    // project_manager._parse_task_deadline. Built as local-time components
    // rather than `new Date(value)` directly: a bare "YYYY-MM-DD" string
    // parses as UTC midnight per spec, which can render as the previous day
    // in negative-UTC-offset timezones.
    const hasTime = value.includes("T");
    const [datePart, timePart] = value.split("T");
    const [y, m, d] = datePart.split("-").map(Number);
    const date = hasTime
      ? new Date(y, m - 1, d, ...timePart.split(":").map(Number))
      : new Date(y, m - 1, d);
    if (isNaN(date.getTime())) return value;
    const opts = { year: "numeric", month: "short", day: "numeric" };
    if (hasTime) {
      opts.hour = "numeric";
      opts.minute = "2-digit";
    }
    return date.toLocaleString(undefined, opts);
  }

  function parseLocalDate(value) {
    // "YYYY-MM-DD" parsed as local midnight (see formatDeadline's note on
    // why `new Date(value)` directly is wrong here).
    if (!value) return null;
    const [y, m, d] = value.split("-").map(Number);
    const date = new Date(y, m - 1, d);
    return isNaN(date.getTime()) ? null : date;
  }

  function formatWeeks(totalDays) {
    const days = Math.round(totalDays);
    const weeks = Math.floor(days / 7);
    const remDays = days % 7;
    const weeksPart = `${weeks} week${weeks === 1 ? "" : "s"}`;
    if (remDays === 0) return weeksPart;
    const daysPart = `${remDays} day${remDays === 1 ? "" : "s"}`;
    return weeks === 0 ? daysPart : `${weeksPart}, ${daysPart}`;
  }

  function projectWeeksInfo(project) {
    const start = parseLocalDate(project.gantt_start);
    const end = parseLocalDate(project.gantt_end);
    if (!start || !end) return { total: "Not set", remaining: "Not set" };
    const msPerDay = 24 * 60 * 60 * 1000;
    const totalDays = Math.max(0, (end - start) / msPerDay);
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    // Before the project starts, "remaining" is the whole project - counting
    // down from today would overstate it since today < start.
    const remainingMs = today < start ? end - start : end - today;
    return {
      total: formatWeeks(totalDays),
      remaining: remainingMs <= 0 ? "0 days (past end date)" : formatWeeks(remainingMs / msPerDay),
    };
  }

  function formatDurationInput(minutes) {
    // Mirrors project_manager._format_duration_tag's "1h30m"/"2h"/"45m"
    // shape - the edit-mode duration input round-trips through this same
    // compact text, and the backend parser (_parse_task_duration, via
    // duration_manager.parse_duration_minutes) accepts it back unchanged.
    const total = Math.round(minutes);
    const hours = Math.floor(total / 60);
    const mins = total % 60;
    if (hours && mins) return `${hours}h${mins}m`;
    if (hours) return `${hours}h`;
    return `${mins}m`;
  }

  function formatTaskProperties(t) {
    // Client-side mirror of project_manager._format_task_properties_inline
    // for the board card's view-mode badge.
    const parts = [];
    if (t.duration_minutes != null) parts.push(`~${formatDurationInput(t.duration_minutes)}`);
    if (t.effort) parts.push(`${t.effort} effort`);
    if (t.when) parts.push(t.when);
    return parts.join(" · ");
  }

  async function loadProjects() {
    const data = await apiFetch("/projects", { method: "GET" });
    projectsById = {};
    projectSelect.innerHTML = "";
    for (const p of data.projects) {
      projectsById[p.id] = p;
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = `${p.short_code} — ${p.name}`;
      projectSelect.appendChild(opt);
    }
    if (projectSelect.value) fillDatesFromSelection();
  }

  function fillDatesFromSelection() {
    const p = projectsById[projectSelect.value];
    startInput.value = (p && p.gantt_start) || "";
    endInput.value = (p && p.gantt_end) || "";
  }

  const newProjectToggle = document.getElementById("newProjectToggle");
  const newProjectModalOverlay = document.getElementById("newProjectModalOverlay");
  const newProjectName = document.getElementById("newProjectName");
  const newProjectParentSelect = document.getElementById("newProjectParentSelect");
  const newProjectChildCheckbox = document.getElementById("newProjectChildCheckbox");
  const newProjectStart = document.getElementById("newProjectStart");
  const newProjectEnd = document.getElementById("newProjectEnd");
  const newProjectCancelBtn = document.getElementById("newProjectCancelBtn");
  const newProjectCreateBtn = document.getElementById("newProjectCreateBtn");

  function openNewProjectModal() {
    newProjectName.value = "";
    newProjectStart.value = "";
    newProjectEnd.value = "";
    newProjectChildCheckbox.checked = false;
    newProjectParentSelect.disabled = true;
    const sorted = Object.values(projectsById).sort((a, b) => a.name.localeCompare(b.name));
    fillOptions(newProjectParentSelect, sorted, {
      value: (p) => p.id,
      label: (p) => `${p.short_code} — ${p.name}`,
    });
    newProjectModalOverlay.hidden = false;
    newProjectName.focus();
  }

  function closeNewProjectModal() {
    newProjectModalOverlay.hidden = true;
  }

  newProjectToggle.addEventListener("click", openNewProjectModal);
  newProjectCancelBtn.addEventListener("click", closeNewProjectModal);
  newProjectModalOverlay.addEventListener("click", (e) => {
    if (e.target === newProjectModalOverlay) closeNewProjectModal();
  });
  newProjectChildCheckbox.addEventListener("change", () => {
    newProjectParentSelect.disabled = !newProjectChildCheckbox.checked;
  });

  newProjectCreateBtn.addEventListener("click", async () => {
    const rawName = newProjectName.value.trim();
    if (!rawName) {
      showStatus("Enter a project name.", false);
      return;
    }
    let finalName = rawName;
    if (newProjectChildCheckbox.checked) {
      const parent = projectsById[newProjectParentSelect.value];
      if (!parent) {
        showStatus("Select a parent project.", false);
        return;
      }
      finalName = `${parent.short_code}-${rawName}`;
    }
    try {
      const project = await apiFetch("/projects", jsonRequest("POST", { name: finalName }));
      if (newProjectStart.value || newProjectEnd.value) {
        await apiFetch(`/projects/${project.id}`, jsonRequest("PATCH", {
          gantt_start: newProjectStart.value,
          gantt_end: newProjectEnd.value,
        }));
      }
      closeNewProjectModal();
      showStatus("Project created.", true);
      await loadProjects();
      await loadGantt();
    } catch (err) {
      showStatus(err.message, false);
    }
  });

  const ganttFilterToggle = document.getElementById("ganttFilterToggle");
  const ganttFilterDropdown = document.getElementById("ganttFilterDropdown");
  const ganttFilterList = document.getElementById("ganttFilterList");
  const ganttFilterShowAll = document.getElementById("ganttFilterShowAll");
  const ganttFilterHideAll = document.getElementById("ganttFilterHideAll");

  function setGanttFilterOpen(open) {
    ganttFilterDropdown.hidden = !open;
    ganttFilterToggle.classList.toggle("active-accent", open);
    ganttFilterToggle.setAttribute("aria-expanded", String(open));
  }

  ganttFilterToggle.addEventListener("click", () => setGanttFilterOpen(ganttFilterDropdown.hidden));
  document.addEventListener("click", (e) => {
    if (ganttFilterDropdown.hidden) return;
    if (e.target === ganttFilterToggle || ganttFilterDropdown.contains(e.target)) return;
    setGanttFilterOpen(false);
  });
  ganttFilterShowAll.addEventListener("click", () => {
    hiddenProjectIds.clear();
    saveHiddenProjectIds();
    renderGanttFilterList();
    renderGanttChart();
  });
  ganttFilterHideAll.addEventListener("click", () => {
    hiddenProjectIds = new Set(allGanttRows.map((r) => r.id));
    saveHiddenProjectIds();
    renderGanttFilterList();
    renderGanttChart();
  });

  async function showProjectInfo(projectId) {
    taskSidebarContent.innerHTML = '<p class="hint">Loading project…</p>';
    try {
      const project = await apiFetch(`/projects/${projectId}`, { method: "GET" }, "Could not load that project.");
      const header = `<h3>${project.short_code} <span class="proj-code">— ${project.name}</span></h3>`;

      let children = [];
      try {
        ({ children } = await apiFetch(`/gantt/${projectId}/children`, { method: "GET" }));
      } catch (e) {
        // Children lookup failing shouldn't block the rest of the info view.
      }

      const childrenHtml = children.length
        ? `<div class="project-info-row">
            <span class="label">Child projects</span>
            <ul class="project-info-children">${children.map((c) => `<li>
              <span class="task-title">${c.short_code} — ${c.name}</span>
              <div class="task-meta"><span class="task-status ${c.status}">${c.status}</span></div>
            </li>`).join("")}</ul>
          </div>`
        : `<div class="project-info-row">
            <span class="label">Child projects</span>
            <span class="value">None</span>
          </div>`;

      const weeksInfo = projectWeeksInfo(project);

      if (!chartEditMode) {
        const rows = [
          { label: "Full name", value: project.name },
          { label: "Status", value: `<span class="task-status ${project.status}">${project.status}</span>` },
          { label: "Start date", value: project.gantt_start ? formatDeadline(project.gantt_start) : "Not set" },
          { label: "End date", value: project.gantt_end ? formatDeadline(project.gantt_end) : "Not set" },
          { label: "Total weeks", value: weeksInfo.total },
          { label: "Weeks remaining", value: weeksInfo.remaining },
        ];
        const rowsHtml = rows.map((r) => `<div class="project-info-row">
          <span class="label">${r.label}</span>
          <span class="value">${r.value}</span>
        </div>`).join("");
        taskSidebarContent.innerHTML = header + `<div class="project-info">${rowsHtml}${childrenHtml}</div>`;
        return;
      }

      taskSidebarContent.innerHTML = header + `<div class="project-info">${childrenHtml}</div>`;
      const infoBox = taskSidebarContent.querySelector(".project-info");

      const nameRow = document.createElement("div");
      nameRow.className = "project-info-row";
      nameRow.innerHTML = '<span class="label">Full name</span>';
      const nameInput = document.createElement("input");
      nameInput.type = "text";
      nameInput.className = "edit-input";
      nameInput.value = project.name;
      nameInput.addEventListener("change", () => saveProjectField(projectId, { name: nameInput.value }));
      nameRow.appendChild(nameInput);

      const statusRow = document.createElement("div");
      statusRow.className = "project-info-row";
      statusRow.innerHTML = '<span class="label">Status</span>';
      const statusSelect = document.createElement("select");
      statusSelect.className = "edit-select";
      fillStatusOptions(statusSelect, PROJECT_STATUSES);
      statusSelect.value = project.status;
      statusSelect.addEventListener("change", () => saveProjectField(projectId, { status: statusSelect.value }));
      statusRow.appendChild(statusSelect);

      const startRow = document.createElement("div");
      startRow.className = "project-info-row";
      startRow.innerHTML = '<span class="label">Start date</span>';
      const startInputEl = document.createElement("input");
      startInputEl.type = "date";
      startInputEl.className = "edit-input";
      startInputEl.value = project.gantt_start || "";
      startInputEl.addEventListener("change", () => saveProjectField(projectId, { gantt_start: startInputEl.value }));
      startRow.appendChild(startInputEl);

      const endRow = document.createElement("div");
      endRow.className = "project-info-row";
      endRow.innerHTML = '<span class="label">End date</span>';
      const endInputEl = document.createElement("input");
      endInputEl.type = "date";
      endInputEl.className = "edit-input";
      endInputEl.value = project.gantt_end || "";
      endInputEl.addEventListener("change", () => saveProjectField(projectId, { gantt_end: endInputEl.value }));
      endRow.appendChild(endInputEl);

      const weeksRow = document.createElement("div");
      weeksRow.className = "project-info-row";
      weeksRow.innerHTML = `<span class="label">Total weeks</span><span class="value">${weeksInfo.total}</span>`;
      const remainingRow = document.createElement("div");
      remainingRow.className = "project-info-row";
      remainingRow.innerHTML = `<span class="label">Weeks remaining</span><span class="value">${weeksInfo.remaining}</span>`;

      const fragment = document.createDocumentFragment();
      fragment.appendChild(nameRow);
      fragment.appendChild(statusRow);
      fragment.appendChild(startRow);
      fragment.appendChild(endRow);
      fragment.appendChild(weeksRow);
      fragment.appendChild(remainingRow);
      infoBox.insertBefore(fragment, infoBox.firstChild);
    } catch (e) {
      taskSidebarContent.innerHTML = `<p class="hint">${e.message}</p>`;
    }
  }

  async function saveProjectField(projectId, patch) {
    try {
      await apiFetch(`/projects/${projectId}`, jsonRequest("PATCH", patch));
      showStatus("Saved.", true);
      await loadProjects();
      await loadGantt();
      await showProjectInfo(projectId);
    } catch (err) {
      showStatus(err.message, false);
    }
  }

  function setChartEditMode(enabled) {
    chartEditMode = enabled;
    chartEditToggle.classList.toggle("active-ok", enabled);
    chartEditToggle.setAttribute("aria-pressed", String(enabled));
    chartEditToggle.title = enabled ? "Editing project (click to lock)" : "Edit project";
    if (boardProjectId) showProjectInfo(boardProjectId);
  }

  chartEditToggle.addEventListener("click", () => setChartEditMode(!chartEditMode));

  async function loadBoard(projectId) {
    boardProjectId = projectId;
    boardHint.hidden = false;
    boardHint.textContent = "Loading board…";
    board.hidden = true;
    try {
      const project = await apiFetch(`/projects/${projectId}`, { method: "GET" }, "Could not load that project.");
      renderBoard(project);
    } catch (e) {
      boardHint.textContent = e.message;
    }
  }

  async function saveTaskField(taskId, patch) {
    try {
      await apiFetch(`/projects/${boardProjectId}/tasks/${taskId}`, jsonRequest("PATCH", patch));
      showStatus("Saved.", true);
    } catch (err) {
      showStatus(err.message, false);
    }
  }

  function setBoardEditMode(enabled) {
    boardEditMode = enabled;
    boardEditToggle.classList.toggle("active-ok", enabled);
    boardEditToggle.setAttribute("aria-pressed", String(enabled));
    boardEditToggle.title = enabled ? "Editing tasks (click to lock)" : "Edit tasks";
    if (enabled) setBoardDeleteMode(false, { skipReload: true });
    if (boardProjectId) loadBoard(boardProjectId);
  }

  function setBoardDeleteMode(enabled, opts) {
    boardDeleteMode = enabled;
    boardDeleteToggle.classList.toggle("active-danger", enabled);
    boardDeleteToggle.setAttribute("aria-pressed", String(enabled));
    boardDeleteToggle.title = enabled ? "Deleting tasks (click to lock)" : "Delete tasks";
    if (enabled) setBoardEditMode(false, { skipReload: true });
    if (!(opts && opts.skipReload) && boardProjectId) loadBoard(boardProjectId);
  }

  boardEditToggle.addEventListener("click", () => setBoardEditMode(!boardEditMode));
  boardDeleteToggle.addEventListener("click", () => setBoardDeleteMode(!boardDeleteMode));

  const boardAddToggle = document.getElementById("boardAddToggle");
  const taskModalOverlay = document.getElementById("taskModalOverlay");
  const newTaskProject = document.getElementById("newTaskProject");
  const newTaskTitle = document.getElementById("newTaskTitle");
  const newTaskStatus = document.getElementById("newTaskStatus");
  const newTaskCancelBtn = document.getElementById("newTaskCancelBtn");
  const newTaskCreateBtn = document.getElementById("newTaskCreateBtn");
  fillStatusOptions(newTaskStatus, TASK_STATUSES);

  function openTaskModal() {
    if (!boardProjectId) {
      showStatus("Select a project on the board first.", false);
      return;
    }
    const project = projectsById[boardProjectId];
    newTaskProject.value = project ? `${project.short_code} — ${project.name}` : boardProjectId;
    newTaskTitle.value = "";
    newTaskStatus.value = "pending";
    taskModalOverlay.hidden = false;
    newTaskTitle.focus();
  }

  function closeTaskModal() {
    taskModalOverlay.hidden = true;
  }

  boardAddToggle.addEventListener("click", openTaskModal);
  newTaskCancelBtn.addEventListener("click", closeTaskModal);
  taskModalOverlay.addEventListener("click", (e) => {
    if (e.target === taskModalOverlay) closeTaskModal();
  });

  newTaskCreateBtn.addEventListener("click", async () => {
    const title = newTaskTitle.value.trim();
    if (!title) {
      showStatus("Enter a task title.", false);
      return;
    }
    try {
      const task = await apiFetch(`/projects/${boardProjectId}/tasks`, jsonRequest("POST", { title }));
      if (newTaskStatus.value !== "pending") {
        await apiFetch(`/projects/${boardProjectId}/tasks/${task.id}`, jsonRequest("PATCH", { status: newTaskStatus.value }));
      }
      closeTaskModal();
      showStatus("Task created.", true);
      await loadBoard(boardProjectId);
    } catch (err) {
      showStatus(err.message, false);
    }
  });

  // Builds the deadline-date + deadline-time input pair used on an
  // editable board card, wired so either input saves the combined
  // "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM" deadline (matching
  // project_manager._parse_task_deadline), and clearing the date clears
  // the whole deadline.
  function buildDeadlineEditor(task) {
    const wrap = document.createElement("div");
    wrap.className = "card-deadline-wrap";
    const dateInput = document.createElement("input");
    dateInput.type = "date";
    dateInput.className = "card-deadline-date-input";
    dateInput.title = "Deadline date";
    const timeInput = document.createElement("input");
    timeInput.type = "time";
    timeInput.className = "card-deadline-time-input";
    timeInput.title = "Deadline time (optional)";
    if (task.deadline) {
      const [datePart, timePart] = task.deadline.split("T");
      dateInput.value = datePart;
      if (timePart) timeInput.value = timePart;
    }
    const save = () => {
      if (!dateInput.value) {
        timeInput.value = "";
        saveTaskField(task.id, { deadline: "" });
        return;
      }
      const deadline = timeInput.value ? `${dateInput.value}T${timeInput.value}` : dateInput.value;
      saveTaskField(task.id, { deadline });
    };
    dateInput.addEventListener("change", save);
    timeInput.addEventListener("change", save);
    wrap.appendChild(dateInput);
    wrap.appendChild(timeInput);
    return wrap;
  }

  // Builds the "when" editor: a time-of-day select + optional
  // weekday/weekend modifier select, combined into the single canonical
  // string project_manager._parse_task_when expects (e.g. "afternoon
  // weekend"). Clearing the time word clears the whole field.
  function buildWhenEditor(task) {
    const wrap = document.createElement("div");
    wrap.className = "card-when-wrap";
    const [timePart, modifierPart] = (task.when || "").split(" ");
    const timeSelect = document.createElement("select");
    timeSelect.className = "card-when-time-select";
    fillOptions(timeSelect, WHEN_TIME_OPTIONS);
    timeSelect.value = timePart || "";
    const modifierSelect = document.createElement("select");
    modifierSelect.className = "card-when-modifier-select";
    fillOptions(modifierSelect, WHEN_MODIFIER_OPTIONS);
    modifierSelect.value = modifierPart || "";
    const save = () => {
      if (!timeSelect.value) {
        modifierSelect.value = "";
        saveTaskField(task.id, { when: "" });
        return;
      }
      const when = modifierSelect.value ? `${timeSelect.value} ${modifierSelect.value}` : timeSelect.value;
      saveTaskField(task.id, { when });
    };
    timeSelect.addEventListener("change", save);
    modifierSelect.addEventListener("change", save);
    wrap.appendChild(timeSelect);
    wrap.appendChild(modifierSelect);
    return wrap;
  }

  function buildEditableCard(t) {
    const card = document.createElement("div");
    card.className = "board-card editing";
    card.dataset.taskId = t.id;
    card.draggable = false;

    const titleInput = document.createElement("input");
    titleInput.type = "text";
    titleInput.className = "card-title-input";
    titleInput.value = t.title;
    titleInput.addEventListener("change", () => saveTaskField(t.id, { title: titleInput.value }));

    const notesInput = document.createElement("textarea");
    notesInput.className = "card-note-input";
    notesInput.placeholder = "Notes";
    notesInput.value = t.notes || "";
    notesInput.addEventListener("change", () => saveTaskField(t.id, { notes: notesInput.value }));

    const priorityLabel = document.createElement("label");
    priorityLabel.className = "card-priority-toggle";
    const priorityCheckbox = document.createElement("input");
    priorityCheckbox.type = "checkbox";
    priorityCheckbox.checked = t.priority === "high";
    priorityCheckbox.addEventListener("change", () => {
      saveTaskField(t.id, { priority: priorityCheckbox.checked ? "high" : "normal" });
    });
    priorityLabel.appendChild(priorityCheckbox);
    priorityLabel.appendChild(document.createTextNode("High priority"));

    // Duration: free text, same shapes project_manager._parse_task_duration
    // accepts ("45m", "1h30m", "90", ...). Empty clears it.
    const durationInput = document.createElement("input");
    durationInput.type = "text";
    durationInput.className = "card-duration-input";
    durationInput.placeholder = "Duration (e.g. 45m)";
    durationInput.value = t.duration_minutes != null ? formatDurationInput(t.duration_minutes) : "";
    durationInput.addEventListener("change", () => saveTaskField(t.id, { duration: durationInput.value }));

    // Effort: select, empty option clears it.
    const effortSelect = document.createElement("select");
    effortSelect.className = "card-effort-select";
    fillOptions(effortSelect, EFFORT_OPTIONS);
    effortSelect.value = t.effort || "";
    effortSelect.addEventListener("change", () => saveTaskField(t.id, { effort: effortSelect.value }));

    card.appendChild(titleInput);
    card.appendChild(notesInput);
    card.appendChild(buildDeadlineEditor(t));
    card.appendChild(durationInput);
    card.appendChild(effortSelect);
    card.appendChild(buildWhenEditor(t));
    card.appendChild(priorityLabel);
    return card;
  }

  function buildViewCard(t) {
    const card = document.createElement("div");
    card.className = "board-card";
    card.dataset.taskId = t.id;

    const flag = t.priority === "high" ? '<span class="card-priority-flag" title="High priority">&#10071;</span>' : "";
    const note = t.notes ? `<span class="card-note">${t.notes}</span>` : "";
    const deadline = t.deadline ? `<span class="card-deadline">Due ${formatDeadline(t.deadline)}</span>` : "";
    const properties = formatTaskProperties(t);
    const propertiesLine = properties ? `<span class="card-properties">${properties}</span>` : "";
    card.innerHTML = `${flag}<span class="card-title">${t.title}</span>${deadline}${propertiesLine}${note}`;

    if (boardDeleteMode) {
      card.draggable = false;
      card.addEventListener("click", async () => {
        if (!confirm(`Delete task "${t.title}"?`)) return;
        try {
          await apiFetch(`/projects/${boardProjectId}/tasks/${t.id}`, { method: "DELETE" });
          showStatus("Task deleted.", true);
          await loadBoard(boardProjectId);
        } catch (err) {
          showStatus(err.message, false);
        }
      });
    } else {
      card.draggable = true;
      card.addEventListener("dragstart", (e) => {
        card.classList.add("dragging");
        e.dataTransfer.setData("text/plain", t.id);
        e.dataTransfer.effectAllowed = "move";
      });
      card.addEventListener("dragend", () => card.classList.remove("dragging"));
    }
    return card;
  }

  function renderBoard(project) {
    const tasks = Object.values(project.tasks || {});
    tasks.sort((a, b) => (a.sort_order || 0) - (b.sort_order || 0));
    board.innerHTML = "";
    for (const s of TASK_STATUSES) {
      const col = document.createElement("div");
      col.className = "board-col";
      col.dataset.status = s.key;

      const header = document.createElement("div");
      header.className = "board-col-header";
      const colTasks = tasks.filter((t) => t.status === s.key);
      header.innerHTML = `<span>${s.label}</span><span class="board-col-count">${colTasks.length}</span>`;
      col.appendChild(header);

      const cards = document.createElement("div");
      cards.className = "board-cards";
      for (const t of colTasks) {
        cards.appendChild(boardEditMode ? buildEditableCard(t) : buildViewCard(t));
      }
      if (boardDeleteMode) cards.classList.add("delete-mode");
      col.appendChild(cards);

      col.addEventListener("dragover", (e) => {
        if (boardEditMode || boardDeleteMode) return;
        e.preventDefault();
        col.classList.add("drag-over");
      });
      col.addEventListener("dragleave", () => col.classList.remove("drag-over"));
      col.addEventListener("drop", async (e) => {
        if (boardEditMode || boardDeleteMode) return;
        e.preventDefault();
        col.classList.remove("drag-over");
        const taskId = e.dataTransfer.getData("text/plain");
        const task = tasks.find((t) => t.id === taskId);
        if (!task || task.status === s.key) return;
        try {
          await apiFetch(`/projects/${boardProjectId}/tasks/${taskId}`, jsonRequest("PATCH", { status: s.key }));
          await loadBoard(boardProjectId);
        } catch (err) {
          showStatus(err.message, false);
        }
      });

      board.appendChild(col);
    }
    boardHint.hidden = true;
    board.hidden = false;
  }

  function toDateInputValue(date) {
    const y = date.getFullYear();
    const m = String(date.getMonth() + 1).padStart(2, "0");
    const d = String(date.getDate()).padStart(2, "0");
    return `${y}-${m}-${d}`;
  }

  function getDragLabel() {
    if (!dragLabelEl) {
      dragLabelEl = document.createElement("div");
      dragLabelEl.className = "bar-drag-label";
      document.body.appendChild(dragLabelEl);
    }
    return dragLabelEl;
  }

  function showDragLabel(handleEl, date) {
    const label = getDragLabel();
    label.textContent = date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
    label.style.display = "block";
    const handleRect = handleEl.getBoundingClientRect();
    const labelRect = label.getBoundingClientRect();
    label.style.left = `${handleRect.left + handleRect.width / 2 - labelRect.width / 2}px`;
    label.style.top = `${handleRect.top - labelRect.height - 8}px`;
  }

  function hideDragLabel() {
    if (dragLabelEl) dragLabelEl.style.display = "none";
  }

  // Track which handle (left/right border) started the current drag so
  // on_date_change below knows which date to show and where to anchor the label.
  document.addEventListener("mousedown", (e) => {
    const handle = e.target.closest(".handle.left, .handle.right");
    if (!handle) return;
    const wrapper = handle.closest(".bar-wrapper");
    if (!wrapper) return;
    activeDrag = {
      taskId: wrapper.getAttribute("data-id"),
      side: handle.classList.contains("left") ? "left" : "right",
    };
  });
  document.addEventListener("mouseup", () => {
    activeDrag = null;
    hideDragLabel();
  });

  // frappe-gantt fires its date_change event on every mousemove while a bar
  // border is being dragged, not just once on release. Debounce so only the
  // final position (settled once the mouse stops, i.e. on release) is saved.
  const pendingBarSaves = {};
  function saveBarDates(task, start, end) {
    if (activeDrag && activeDrag.taskId === task.id) {
      const wrapper = document.querySelector(`.bar-wrapper[data-id="${task.id}"]`);
      const handleEl = wrapper && wrapper.querySelector(`.handle.${activeDrag.side}`);
      if (handleEl) showDragLabel(handleEl, activeDrag.side === "left" ? start : end);
    }
    clearTimeout(pendingBarSaves[task.id]);
    pendingBarSaves[task.id] = setTimeout(async () => {
      try {
        await apiFetch(`/projects/${task.id}`, jsonRequest("PATCH", {
          gantt_start: toDateInputValue(start),
          gantt_end: toDateInputValue(end),
        }));
        showStatus("Saved.", true);
        await loadProjects();
      } catch (e) {
        showStatus(e.message, false);
        await loadGantt();
      }
    }, 300);
  }

  let allGanttRows = [];
  let hiddenProjectIds = new Set();
  try {
    const saved = JSON.parse(localStorage.getItem("gantt.hiddenProjectIds") || "[]");
    hiddenProjectIds = new Set(Array.isArray(saved) ? saved : []);
  } catch (e) {}

  function saveHiddenProjectIds() {
    try { localStorage.setItem("gantt.hiddenProjectIds", JSON.stringify(Array.from(hiddenProjectIds))); } catch (e) {}
  }

  function renderGanttChart() {
    const rows = allGanttRows.filter((r) => !hiddenProjectIds.has(r.id));
    if (!allGanttRows.length) {
      ganttEmpty.textContent = "No projects have both a start and end date set yet — use the form above to add one.";
      ganttEmpty.hidden = false;
      chartArea.style.display = "none";
      return;
    }
    if (!rows.length) {
      ganttEmpty.textContent = "All projects are hidden by the current filter.";
      ganttEmpty.hidden = false;
      chartArea.style.display = "none";
      return;
    }
    ganttEmpty.hidden = true;
    chartArea.style.display = "flex";
    if (ganttInstance) {
      ganttInstance.refresh(rows);
    } else {
      ganttInstance = new Gantt("#gantt", rows, {
        view_mode: currentViewMode,
        date_format: "YYYY-MM-DD",
        readonly_progress: true,
        on_click: (task) => {
          document.querySelectorAll(".bar-wrapper.selected").forEach((el) => el.classList.remove("selected"));
          const wrapper = document.querySelector(`.bar-wrapper[data-id="${task.id}"]`);
          if (wrapper) wrapper.classList.add("selected");
          showProjectInfo(task.id);
          loadBoard(task.id);
        },
        on_date_change: (task, start, end) => saveBarDates(task, start, end),
      });
    }
  }

  function renderGanttFilterList() {
    ganttFilterList.innerHTML = "";
    for (const row of allGanttRows) {
      const label = document.createElement("label");
      label.className = "filter-dropdown-item";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = !hiddenProjectIds.has(row.id);
      checkbox.addEventListener("change", () => {
        if (checkbox.checked) hiddenProjectIds.delete(row.id);
        else hiddenProjectIds.add(row.id);
        saveHiddenProjectIds();
        renderGanttChart();
      });
      label.appendChild(checkbox);
      label.appendChild(document.createTextNode(row.name));
      ganttFilterList.appendChild(label);
    }
  }

  async function loadGantt() {
    const data = await apiFetch("/gantt", { method: "GET" });
    allGanttRows = data.rows || [];
    const knownIds = new Set(allGanttRows.map((r) => r.id));
    let pruned = false;
    for (const id of Array.from(hiddenProjectIds)) {
      if (!knownIds.has(id)) {
        hiddenProjectIds.delete(id);
        pruned = true;
      }
    }
    if (pruned) saveHiddenProjectIds();
    renderGanttFilterList();
    renderGanttChart();
  }

  function setDatesPanelCollapsed(collapsed) {
    datesBody.hidden = collapsed;
    datesToggle.setAttribute("aria-expanded", String(!collapsed));
    try { localStorage.setItem("gantt.datesPanelCollapsed", collapsed ? "1" : "0"); } catch (e) {}
  }

  datesToggle.addEventListener("click", () => {
    setDatesPanelCollapsed(!datesBody.hidden);
  });

  function setSidebarCollapsed(collapsed) {
    taskSidebar.classList.toggle("collapsed", collapsed);
    sidebarToggle.textContent = collapsed ? "▶" : "◀";
    sidebarToggle.title = collapsed ? "Expand" : "Collapse";
    try { localStorage.setItem("gantt.sidebarCollapsed", collapsed ? "1" : "0"); } catch (e) {}
  }

  sidebarToggle.addEventListener("click", () => {
    setSidebarCollapsed(!taskSidebar.classList.contains("collapsed"));
  });

  projectSelect.addEventListener("change", fillDatesFromSelection);
  zoomOutBtn.addEventListener("click", () => setViewMode("Month"));
  zoomInBtn.addEventListener("click", () => setViewMode("Week"));

  document.getElementById("saveBtn").addEventListener("click", async () => {
    const projectId = projectSelect.value;
    if (!projectId) return;
    const body = {};
    if (startInput.value) body.gantt_start = startInput.value;
    if (endInput.value) body.gantt_end = endInput.value;
    if (!body.gantt_start && !body.gantt_end) {
      showStatus("Set a start and/or end date first.", false);
      return;
    }
    try {
      await apiFetch(`/projects/${projectId}`, jsonRequest("PATCH", body));
      showStatus("Saved.", true);
      await loadProjects();
      await loadGantt();
    } catch (e) {
      showStatus(e.message, false);
    }
  });

  document.getElementById("clearBtn").addEventListener("click", async () => {
    const projectId = projectSelect.value;
    if (!projectId) return;
    try {
      await apiFetch(`/projects/${projectId}`, jsonRequest("PATCH", { clear_gantt_dates: true }));
      showStatus("Cleared.", true);
      startInput.value = "";
      endInput.value = "";
      await loadProjects();
      await loadGantt();
    } catch (e) {
      showStatus(e.message, false);
    }
  });

  (async function init() {
    let collapsed = false;
    try { collapsed = localStorage.getItem("gantt.datesPanelCollapsed") === "1"; } catch (e) {}
    setDatesPanelCollapsed(collapsed);
    let sidebarCollapsed = false;
    try { sidebarCollapsed = localStorage.getItem("gantt.sidebarCollapsed") === "1"; } catch (e) {}
    setSidebarCollapsed(sidebarCollapsed);
    zoomOutBtn.classList.toggle("active", currentViewMode === "Month");
    zoomInBtn.classList.toggle("active", currentViewMode === "Week");
    await loadProjects();
    await loadGantt();
  })();
})();
