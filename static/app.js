
"use strict";

// ---------------------------------------------
// State and helpers
// ---------------------------------------------

const state = {
  documentId: null,
  sessionId: null,
  documents: [],
  sessions: [],
  rubrics: [],
  busy: false,
  reportIds: new Set()
};

const $ = (selector) => document.querySelector(selector);

function element(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.textContent = text;
  return node;
}

function clear(node) {
  node.replaceChildren();
}

function notify(message, error = false) {
  const box = $("#notification");
  box.textContent = message;
  box.className = error
    ? "notification error"
    : "notification";
}

function setBusy(busy) {
  state.busy = busy;

  for (const id of [
    "send-button",
    "new-session-button",
    "grade-button"
  ]) {
    $( "#" + id ).disabled = busy;
  }
}


//
// V11 — CSRF-aware API helper
//

async function getCsrfToken() {
  const response = await fetch("/auth/csrf", {
    method: "GET",
    credentials: "same-origin",
    cache: "no-store"
  });

  if (!response.ok) {
    if (response.status === 401) {
      throw new Error(
        "Your login session has expired. Please sign in again."
      );
    }

    throw new Error(
      `Unable to obtain CSRF token (HTTP ${response.status}).`
    );
  }

  const data = await response.json();

  if (
    typeof data.csrf_token !== "string" ||
    !data.csrf_token
  ) {
    throw new Error(
      "The server returned an invalid CSRF token."
    );
  }

  return data.csrf_token;
}


async function api(path, options = {}) {
  const method = (options.method || "GET").toUpperCase();

  const unsafeMethods = [
    "POST",
    "PUT",
    "PATCH",
    "DELETE"
  ];

  // Preserve the existing request headers.
  const headers = new Headers(options.headers || {});

  // Login is exempt because the user does not
  // have an authenticated session yet.
  const needsCsrf =
    unsafeMethods.includes(method) &&
    path !== "/auth/login";

  if (needsCsrf) {
    const csrfToken = await getCsrfToken();

    headers.set(
      "X-CSRF-Token",
      csrfToken
    );
  }

  // Cookies are sent automatically to our FastAPI
  // application on the same origin.
  const response = await fetch(path, {
    ...options,
    credentials: "same-origin",
    headers
  });

  let data;

  try {
    data = await response.json();
  } catch {
    throw new Error(
      `Unexpected server response (HTTP ${response.status}).`
    );
  }

  if (!response.ok) {
    let detail = data.detail || "Request failed.";

    if (typeof detail !== "string") {
      detail = JSON.stringify(detail);
    }

    throw new Error(detail);
  }

  return data;
}


function jsonOptions(data) {
  return {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(data)
  };
}

function addOption(select, value, label) {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = label;
  select.append(option);
}

function shortDate(value) {
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString();
}

// ---------------------------------------------
// Navigation
// ---------------------------------------------

const viewNames = {
  chat: ["Chat with PDF", "Explore research papers with AI."],
  grading: ["Essay Grading", "Assess essays using academic rubrics."],
  saved: ["Saved Work", "Continue conversations and view reports."]
};

function switchView(view) {
  document.querySelectorAll(".view").forEach((node) => {
    node.classList.toggle(
      "active",
      node.id === `view-${view}`
    );
  });

  document.querySelectorAll(".nav-button").forEach((node) => {
    node.classList.toggle(
      "active",
      node.dataset.view === view
    );
  });

  $("#page-title").textContent = viewNames[view][0];
  $("#page-subtitle").textContent = viewNames[view][1];

  if (view === "saved") {
    renderSavedWork();
  }
}

document.querySelectorAll(".nav-button").forEach((button) => {
  button.addEventListener("click", () => {
    switchView(button.dataset.view);
  });
});

// ---------------------------------------------
// Documents
// ---------------------------------------------

async function loadDocuments() {
  const data = await api("/documents");

  state.documents = data.documents || [];

  const select = $("#document-select");
  const previous = state.documentId;

  clear(select);
  addOption(select, "", "Select document");

  for (const doc of state.documents) {
    addOption(
      select,
      doc.document_id,
      `${doc.filename} (${doc.page_count} pages)`
    );
  }

  if (previous && state.documents.some(
    d => d.document_id === previous
  )) {
    select.value = previous;
  }
}

function selectedDocument() {
  return state.documents.find(
    d => d.document_id === state.documentId
  );
}

function updateDocumentHeading() {
  const doc = selectedDocument();

  $("#chat-document-name").textContent = doc
    ? doc.filename
    : "Select a PDF to begin.";
}

$("#document-select").addEventListener("change", (event) => {
  state.documentId = event.target.value || null;
  state.sessionId = null;

  updateDocumentHeading();
  showEmptyChat();

  $("#chat-status").textContent = "New conversation";
});

$("#pdf-upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = $("#pdf-file").files[0];

  if (!file) {
    notify("Choose a PDF file.", true);
    return;
  }

  if (file.size > 20 * 1024 * 1024) {
    notify("PDF exceeds the 20 MB limit.", true);
    return;
  }

  if (state.busy) return;

  setBusy(true);

  try {
    $("#upload-status").textContent = "Uploading PDF...";

    const form = new FormData();
    form.append("file", file);

    const uploaded = await api("/upload-pdf", {
      method: "POST",
      body: form
    });

    $("#upload-status").textContent =
      "Preparing chunks, embeddings, and FAISS...";

    await api(
      "/prepare-document",
      jsonOptions({
        document_id: uploaded.document_id,
        chunk_size: 700,
        chunk_overlap: 100
      })
    );

    await loadDocuments();

    state.documentId = uploaded.document_id;
    state.sessionId = null;

    $("#document-select").value = state.documentId;
    updateDocumentHeading();
    showEmptyChat();

    $("#upload-status").textContent = "PDF ready.";
    notify("PDF uploaded and prepared successfully.");

    await createNewSession();

  } catch (error) {
    $("#upload-status").textContent = "Preparation failed.";
    notify(error.message, true);
  } finally {
    setBusy(false);
  }
});

// ---------------------------------------------
// PDF conversation sessions
// ---------------------------------------------

async function loadSessions() {
  const data = await api("/pdf-sessions");
  state.sessions = data.sessions || [];
  renderSessions();
}

function renderSessions() {
  const container = $("#recent-sessions");
  clear(container);

  const sessions = state.sessions
    .filter(s => !state.documentId ||
      s.document_id === state.documentId)
    .slice(0, 15);

  if (!sessions.length) {
    container.append(
      element("p", "muted small", "No conversations yet.")
    );
    return;
  }

  for (const session of sessions) {
    const item = element("div", "session-item");

    const button = element(
      "button",
      "session-open",
      session.title
    );

    button.type = "button";
    button.addEventListener("click", () => {
      openSession(session.session_id);
    });

    item.append(button);
    container.append(item);
  }
}

async function createNewSession() {
  if (!state.documentId) {
    notify("Select a PDF first.", true);
    return;
  }

  const session = await api(
    "/pdf-sessions",
    jsonOptions({
      document_id: state.documentId,
      title: selectedDocument()?.filename ||
             "PDF Conversation"
    })
  );

  state.sessionId = session.session_id;

  await loadSessions();

  showEmptyChat();
  $("#chat-status").textContent = "Ready";
  notify("New PDF conversation created.");
}

$("#new-session-button").addEventListener("click", async () => {
  if (state.busy) return;

  try {
    setBusy(true);
    await createNewSession();
  } catch (error) {
    notify(error.message, true);
  } finally {
    setBusy(false);
  }
});

function showEmptyChat() {
  const messages = $("#messages");
  clear(messages);

  const empty = element("div", "empty-state");
  const icon = element("div", "empty-icon", "✦");
  const title = element(
    "h3", "", "Ask your research paper anything"
  );
  const subtitle = element(
    "p", "",
    "Ask about methodology, findings, definitions, or conclusions."
  );

  empty.append(icon, title, subtitle);
  messages.append(empty);
}

function addMessage(role, text, sources = [], status = "") {
  const messages = $("#messages");

  const empty = messages.querySelector(".empty-state");
  if (empty) empty.remove();

  const row = element("div", `message ${role}`);
  const bubble = element("div", "bubble", text);

  if (role === "assistant" && status) {
    bubble.append(
      element("div", "message-meta", `Status: ${status}`)
    );
  }

  if (role === "assistant" && sources.length) {
    const sourceList = element("div", "source-list");

    for (const source of sources) {
      const details = element("details");
      const summary = element(
        "summary",
        "",
        `${source.source_id || "Source"} · Page ${source.page_number}`
      );
      const passage = element(
        "pre", "", source.text || ""
      );

      details.append(summary, passage);
      sourceList.append(details);
    }

    bubble.append(sourceList);
  }

  row.append(bubble);
  messages.append(row);
  messages.scrollTop = messages.scrollHeight;
}

async function openSession(sessionId) {
  if (state.busy) return;

  try {
    const session = await api(
      `/pdf-sessions/${encodeURIComponent(sessionId)}`
    );

    state.sessionId = session.session_id;
    state.documentId = session.document_id;

    $("#document-select").value = state.documentId;
    updateDocumentHeading();

    clear($("#messages"));

    if (!session.messages.length) {
      showEmptyChat();
    }

    for (const turn of session.messages) {
      addMessage("user", turn.question);
      addMessage(
        "assistant",
        turn.answer,
        turn.sources || [],
        turn.status
      );
    }

    $("#chat-status").textContent = "Session loaded";
    switchView("chat");

  } catch (error) {
    notify(error.message, true);
  }
}

$("#chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = $("#chat-input").value.trim();

  if (!question || state.busy) return;

  if (!state.documentId) {
    notify("Select a PDF first.", true);
    return;
  }

  setBusy(true);

  try {
    if (!state.sessionId) {
      await createNewSession();
    }

    addMessage("user", question);
    $("#chat-input").value = "";

    $("#chat-status").textContent = "Thinking...";

    const result = await api(
      "/pdf-chat-memory",
      jsonOptions({
        session_id: state.sessionId,
        question,
        top_k: 5
      })
    );

    addMessage(
      "assistant",
      result.answer,
      result.sources || [],
      result.status
    );

    $("#chat-status").textContent = "Ready";
    await loadSessions();

  } catch (error) {
    $("#chat-status").textContent = "Error";
    notify(error.message, true);

    // Keep the original question available for retry.
    $("#chat-input").value = question;
  } finally {
    setBusy(false);
  }
});

// ---------------------------------------------
// Rubric editor
// ---------------------------------------------

let criterionCounter = 0;

function addCriterion(values = {}) {
  criterionCounter += 1;

  const card = element("div", "criterion-card");

  const heading = element("div", "criterion-heading");
  heading.append(
    element("strong", "", `Criterion ${criterionCounter}`)
  );

  const remove = element("button", "button secondary", "Remove");
  remove.type = "button";

  remove.addEventListener("click", () => {
    card.remove();
    updateWeightTotal();
  });

  heading.append(remove);
  card.append(heading);

  const fields = [
    ["ID", "criterion-id", values.criterion_id || ""],
    ["Name", "criterion-name", values.name || ""],
    ["Description", "criterion-description",
      values.description || ""],
    ["Weight (%)", "criterion-weight",
      values.weight ?? 25],
    ["Maximum score", "criterion-max",
      values.max_score ?? 10]
  ];

  for (const [labelText, className, value] of fields) {
    const label = element("label", "", labelText);
    let input;

    if (className === "criterion-description") {
      input = element("textarea", className);
      input.rows = 3;
    } else {
      input = element("input", className);
      input.type = className.includes("weight") ||
                   className.includes("max")
        ? "number"
        : "text";
    }

    input.value = value;
    input.required = true;

    if (input.type === "number") {
      input.step = "0.01";
      input.min = className.includes("weight")
        ? "0.01"
        : "0.01";
      input.max = "100";
    }

    if (className === "criterion-id") {
      input.pattern = "[a-zA-Z0-9_-]+";
    }

    input.addEventListener("input", updateWeightTotal);

    label.append(input);
    card.append(label);
  }

  $("#criteria-list").append(card);
  updateWeightTotal();
}

function updateWeightTotal() {
  const weights = [
    ...document.querySelectorAll(".criterion-weight")
  ].map(input => Number(input.value) || 0);

  const total = weights.reduce((a, b) => a + b, 0);

  $("#weight-total").textContent =
    `Total weight: ${total.toFixed(2)}%`;
}

$("#add-criterion").addEventListener("click", () => {
  addCriterion({
    criterion_id: `criterion_${criterionCounter + 1}`
  });
});

function collectRubric() {
  const criteria = [
    ...document.querySelectorAll(".criterion-card")
  ].map(card => ({
    criterion_id:
      card.querySelector(".criterion-id").value.trim(),
    name:
      card.querySelector(".criterion-name").value.trim(),
    description:
      card.querySelector(".criterion-description").value.trim(),
    weight:
      Number(card.querySelector(".criterion-weight").value),
    max_score:
      Number(card.querySelector(".criterion-max").value)
  }));

  const total = criteria.reduce(
    (sum, criterion) => sum + criterion.weight,
    0
  );

  if (Math.abs(total - 100) > 0.001) {
    throw new Error("Rubric weights must total 100%.");
  }

  return {
    title: $("#rubric-title").value.trim(),
    instructions: $("#rubric-instructions").value.trim(),
    criteria
  };
}

async function loadRubrics() {
  const result = await api("/rubrics");
  state.rubrics = result.rubrics || [];

  const select = $("#grading-rubric-select");
  const previous = select.value;

  clear(select);
  addOption(select, "", "Choose rubric");

  for (const rubric of state.rubrics) {
    addOption(
      select,
      rubric.rubric_id,
      rubric.title
    );
  }

  if (state.rubrics.some(r => r.rubric_id === previous)) {
    select.value = previous;
  }
}

$("#rubric-form").addEventListener("submit", async (event) => {
  event.preventDefault();

  try {
    const rubric = collectRubric();

    const saved = await api(
      "/rubrics",
      jsonOptions(rubric)
    );

    await loadRubrics();

    $("#grading-rubric-select").value = saved.rubric_id;

    notify("Rubric saved successfully.");

  } catch (error) {
    notify(error.message, true);
  }
});

// ---------------------------------------------
// Essay submission and grading
// ---------------------------------------------

$("#essay-form").addEventListener("submit", async (event) => {
  event.preventDefault();

  if (state.busy) return;

  const rubricId = $("#grading-rubric-select").value;
  const title = $("#essay-title").value.trim();
  const file = $("#essay-file").files[0];
  const text = $("#essay-text").value.trim();

  if (!rubricId) {
    notify("Select a grading rubric.", true);
    return;
  }

  if (!file && !text) {
    notify("Upload an essay or paste essay text.", true);
    return;
  }

  setBusy(true);

  try {
    let essay;

    if (file) {
      if (file.size > 5 * 1024 * 1024) {
        throw new Error("Essay exceeds the 5 MB limit.");
      }

      const form = new FormData();
      form.append("file", file);
      form.append("title", title);

      essay = await api("/essays/upload", {
        method: "POST",
        body: form
      });

    } else {
      essay = await api(
        "/essays/text",
        jsonOptions({ title, text })
      );
    }

    notify("Essay saved. Generating grading report...");

    const grade = await api(
      "/grade-essay",
      jsonOptions({
        essay_id: essay.essay_id,
        rubric_id: rubricId
      })
    );

    rememberGrade(grade.grade_id);

    renderGrade(
      grade,
      $("#grade-report-content")
    );

    $("#grade-report").classList.remove("hidden");

    notify("Grading report generated.");

    $("#grade-report").scrollIntoView({
      behavior: "smooth",
      block: "start"
    });

  } catch (error) {
    notify(error.message, true);
  } finally {
    setBusy(false);
  }
});

// ---------------------------------------------
// Report rendering
// ---------------------------------------------

function rememberGrade(gradeId) {
  state.reportIds.add(gradeId);

  try {
    localStorage.setItem(
      "chat_scholar_grade_ids",
      JSON.stringify([...state.reportIds])
    );
  } catch {
    // Reports remain accessible using their UUID.
  }
}

function renderGrade(grade, container) {
  clear(container);

  container.append(
    element("p", "muted", grade.essay_title || "")
  );

  container.append(
    element(
      "div",
      "grade-score",
      `${Number(grade.percentage).toFixed(2)}%`
    )
  );

  const progress = element("div", "progress");
  const fill = element("div", "progress-fill");

  fill.style.width = `${
    Math.max(0, Math.min(100, grade.percentage))
  }%`;

  progress.append(fill);
  container.append(progress);

  for (const criterion of grade.criteria || []) {
    const section = element("div", "report-criterion");

    section.append(
      element("h3", "", criterion.name)
    );

    section.append(
      element(
        "p",
        "",
        `${criterion.score}/${criterion.max_score} points · ` +
        `Weight: ${criterion.weight}% · ` +
        `Weighted: ${criterion.weighted_points}`
      )
    );

    section.append(
      element("p", "", criterion.justification)
    );

    for (const quote of criterion.evidence_quotes || []) {
      section.append(
        element("blockquote", "", quote)
      );
    }

    section.append(
      element(
        "p",
        "",
        `Suggested improvement: ${criterion.improvement}`
      )
    );

    container.append(section);
  }

  container.append(
    element("h3", "", "Overall feedback"),
    element("p", "", grade.overall_feedback || "")
  );

  container.append(
    element("h3", "", "Strengths")
  );

  const strengths = element("ul");

  for (const item of grade.strengths || []) {
    strengths.append(element("li", "", item));
  }

  container.append(strengths);

  container.append(
    element("h3", "", "Recommended improvements")
  );

  const improvements = element("ul");

  for (const item of grade.improvements || []) {
    improvements.append(element("li", "", item));
  }

  container.append(improvements);

  container.append(
    element(
      "p",
      "muted small",
      grade.grading_notice ||
      "AI-generated assessment; human review required."
    )
  );

  
  // Show V9 evidence verification results.
  const evidence = grade.evidence_verification;

  if (evidence && typeof evidence === "object") {
    const audit = element("div", "report-criterion");

    audit.append(
      element("h3", "", "Evidence verification")
    );

    audit.append(
      element(
        "p",
        "",
        `Verified quotations: ${evidence.verified_quotes} · ` +
        `Removed quotations: ${evidence.removed_quotes}`
      )
    );

    for (const warning of evidence.warnings || []) {
      audit.append(
        element("p", "muted small", warning)
      );
    }

    audit.append(
      element("p", "muted small", evidence.note || "")
    );

    container.append(audit);
  }

  container.append(
    element(
      "p",
      "muted small",
      `Report ID: ${grade.grade_id}`
    )
  );
}

async function openGrade(gradeId) {
  try {
    const grade = await api(
      `/grades/${encodeURIComponent(gradeId)}`
    );

    rememberGrade(gradeId);
    renderGrade(grade, $("#saved-report-content"));

    $("#saved-report-detail").classList.remove("hidden");

    switchView("saved");

    $("#saved-report-detail").scrollIntoView({
      behavior: "smooth",
      block: "start"
    });

  } catch (error) {
    notify(error.message, true);
  }
}

// ---------------------------------------------
// Saved sessions and reports
// ---------------------------------------------

function renderSavedWork() {
  const sessionContainer = $("#saved-sessions");
  const reportContainer = $("#saved-reports");

  clear(sessionContainer);
  clear(reportContainer);

  if (!state.sessions.length) {
    sessionContainer.append(
      element("p", "muted", "No saved conversations.")
    );
  }

  for (const session of state.sessions) {
    const card = element("div", "saved-card");

    card.append(
      element("strong", "", session.title),
      element(
        "p",
        "muted small",
        `${session.total_turns} turns · ${shortDate(session.updated_at)}`
      )
    );

    const open = element(
      "button",
      "button secondary small-button",
      "Open conversation"
    );

    open.addEventListener("click", () => {
      openSession(session.session_id);
    });

    const remove = element(
      "button",
      "button secondary small-button danger",
      "Delete"
    );

    remove.style.marginLeft = "8px";

    remove.addEventListener("click", async () => {
      if (!confirm(
        "Delete this conversation and all its messages?"
      )) return;

      try {
        await api(
          `/pdf-sessions/${encodeURIComponent(session.session_id)}`,
          { method: "DELETE" }
        );

        if (state.sessionId === session.session_id) {
          state.sessionId = null;
          showEmptyChat();
        }

        await loadSessions();
        renderSavedWork();

        notify("Conversation deleted.");

      } catch (error) {
        notify(error.message, true);
      }
    });

    card.append(open, remove);
    sessionContainer.append(card);
  }

  if (!state.reportIds.size) {
    reportContainer.append(
      element(
        "p",
        "muted",
        "No reports recorded in this browser."
      )
    );
  }

  for (const id of state.reportIds) {
    const card = element("div", "saved-card");

    card.append(
      element("strong", "", `Report ${id.slice(0, 8)}`)
    );

    const button = element(
      "button",
      "button secondary small-button",
      "View report"
    );

    button.addEventListener("click", () => {
      openGrade(id);
    });

    card.append(button);
    reportContainer.append(card);
  }
}

$("#refresh-sessions").addEventListener("click", async () => {
  try {
    await loadSessions();
    renderSavedWork();
    notify("Saved sessions refreshed.");
  } catch (error) {
    notify(error.message, true);
  }
});

$("#open-grade-form").addEventListener("submit", (event) => {
  event.preventDefault();

  const id = $("#grade-id-input").value.trim();

  if (id) openGrade(id);
});

// ---------------------------------------------
// Initialize application
// ---------------------------------------------

async function initialize() {
  try {
    const stored = JSON.parse(
      localStorage.getItem("chat_scholar_grade_ids") || "[]"
    );

    if (Array.isArray(stored)) {
      for (const id of stored) {
        if (typeof id === "string") {
          state.reportIds.add(id);
        }
      }
    }
  } catch {
    // Browser storage may be unavailable.
  }

  addCriterion({
    criterion_id: "argument",
    name: "Argument and Thesis",
    description: "Assess clarity and development of the argument.",
    weight: 30,
    max_score: 10
  });

  addCriterion({
    criterion_id: "evidence",
    name: "Evidence and Analysis",
    description: "Assess evidence quality and critical analysis.",
    weight: 30,
    max_score: 10
  });

  addCriterion({
    criterion_id: "structure",
    name: "Structure",
    description: "Assess organization and logical progression.",
    weight: 20,
    max_score: 10
  });

  addCriterion({
    criterion_id: "language",
    name: "Academic Language",
    description: "Assess grammar, clarity and academic tone.",
    weight: 20,
    max_score: 10
  });

  try {
    await Promise.all([
      loadDocuments(),
      loadSessions(),
      loadRubrics()
    ]);
  } catch (error) {
    notify(
      "Could not load application data: " + error.message,
      true
    );
  }
}


//
// V11 — Browser Authentication
//

let applicationInitialized = false;

function showLoginScreen(message = "") {
  const screen = $("#auth-screen");
  const dashboard = $(".app-shell");
  const errorBox = $("#login-error");

  dashboard.style.display = "none";

  screen.hidden = false;
  screen.style.display = "flex";
  errorBox.textContent = message;
  errorBox.classList.toggle("hidden", !message);
}

async function showAuthenticatedDashboard() {
  const screen = $("#auth-screen");
  const dashboard = $(".app-shell");
  const errorBox = $("#login-error");

  screen.hidden = true;
  screen.style.display = "none";

  dashboard.style.display = "";

  errorBox.textContent = "";
  errorBox.classList.add("hidden");

  // Initialize V10 only after successful authentication.
  if (!applicationInitialized) {
    applicationInitialized = true;
    await initialize();
  }
}

async function checkAuthentication() {
  try {
    const response = await fetch("/auth/me", {
      method: "GET",
      credentials: "same-origin",
      cache: "no-store"
    });

    if (response.status === 200) {
      await showAuthenticatedDashboard();
      return;
    }

    if (response.status === 401) {
      showLoginScreen();
      return;
    }

    showLoginScreen(
      "Unable to verify your session. Please try again."
    );

  } catch (error) {
    showLoginScreen(
      "Cannot connect to the authentication server."
    );
  }
}

$("#login-form").addEventListener("submit", async (event) => {
  event.preventDefault();

  const passwordInput = $("#login-password");
  const loginButton = $("#login-button");
  const password = passwordInput.value;

  if (!password) return;

  loginButton.disabled = true;
  loginButton.textContent = "Signing in...";

  try {
    await api(
      "/auth/login",
      jsonOptions({ password })
    );

    // Clear the password field after the request.
    passwordInput.value = "";

    await showAuthenticatedDashboard();

  } catch (error) {
    passwordInput.value = "";

    showLoginScreen(
      error.message === "Invalid credentials"
        ? "Incorrect password. Please try again."
        : "Login failed. Please try again."
    );

  } finally {
    loginButton.disabled = false;
    loginButton.textContent = "Sign In";
  }
});


// ---------------------------------------------
// V11 — Logout
// ---------------------------------------------

$("#logout-button").addEventListener("click", async () => {
  const logoutButton = $("#logout-button");

  if (logoutButton.disabled) return;

  logoutButton.disabled = true;
  logoutButton.textContent = "Signing out...";

  try {
    // Ask FastAPI to clear the authentication cookie.
    await api("/auth/logout", {
      method: "POST",
      credentials: "same-origin"
    });

    // Reload to reset in-memory documents,
    // conversations, rubrics and other UI state.
    // checkAuthentication() will then show login.
    window.location.reload();

  } catch (error) {
    notify("Logout failed: " + error.message, true);

    logoutButton.disabled = false;
    logoutButton.textContent = "Logout";
  }
});

// Start authentication before initializing V10.
checkAuthentication();



