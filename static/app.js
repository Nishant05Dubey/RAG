// Blend RAG Assistant: calls the real backend (POST /api/ask, GET /api/health).
// All document text is inserted with textContent, never as HTML.

const $ = (id) => document.getElementById(id);
const REQUEST_TIMEOUT_MS = 200000; // a little above the server's Ollama timeout
let pendingAction = null;

function renderAction(action) {
  pendingAction = action.status === "pending" ? action.id : null;
  $("action-panel").hidden = false;
  $("action-status").textContent = action.status;
  $("action-message").textContent = action.message;
  const r = action.request;
  $("action-details").textContent = `${r.leave_type} leave · ${r.start_date} to ${r.end_date} · ${r.calendar_days} calendar day(s)` +
    (r.reason ? ` · Reason: ${r.reason}` : "");
  $("action-buttons").hidden = !pendingAction;
  $("ask-btn").disabled = !!pendingAction;
}

async function decideAction(decision) {
  if (!pendingAction) return;
  $("confirm-action").disabled = $("cancel-action").disabled = true;
  $("error").hidden = true;
  try {
    const response = await fetch("/api/action/decision", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({action_id: pendingAction, decision}),
    });
    const data = await response.json();
    if (!response.ok) {
      showError("Action not executed", data.error.message);
      if ([404, 409].includes(response.status)) {
        pendingAction = null;
        $("action-buttons").hidden = true;
        $("action-status").textContent = "closed";
        $("action-message").textContent = data.error.message;
        $("ask-btn").disabled = false;
      }
      return;
    }
    renderAction(data.action);
  } catch (error) {
    showError("Could not confirm the result", "Retry the same button. A repeated confirmation will not execute the simulation twice.");
  } finally {
    $("confirm-action").disabled = $("cancel-action").disabled = false;
  }
}

$("confirm-action").addEventListener("click", () => decideAction("confirm"));
$("cancel-action").addEventListener("click", () => decideAction("cancel"));

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function pages(list) {
  return (list.length > 1 ? "pages " : "page ") + list.join(", ");
}

// ---- health (left panel) ---------------------------------------------------
async function loadHealth() {
  try {
    const res = await fetch("/api/health");
    const h = await res.json();
    $("status-doc").textContent = h.chunks.ok
      ? `${h.chunks.document || "Policy chunks"}${h.chunks.num_chunks ? ` (${h.chunks.num_chunks} chunks)` : ""}`
      : "Chunks missing";
    $("status-model").textContent = h.ollama.model;
    $("status-ollama").textContent = h.ollama.ok ? "Connected" : (h.ollama.reachable ? "Model not installed" : "Not running");
    $("status-dot").className = "dot " + (h.ollama.ok ? "ok" : "bad");
    if (!h.ok) showError("Setup needed", h.ollama.message || h.chunks.message);
  } catch (e) {
    $("status-ollama").textContent = "Unknown";
    $("status-dot").className = "dot bad";
    showError("Cannot reach the app server", "Is 'python run_rag.py web' still running?");
  }
}

// ---- rendering -------------------------------------------------------------
function showError(title, message) {
  const box = $("error");
  box.replaceChildren(el("strong", "", title), el("span", "", message || ""));
  box.hidden = false;
}

function render(data) {
  const d = data.debug;
  const answered = d.status === "answered";
  $("answer").textContent = data.answer;
  const pill = $("answer-status");
  pill.textContent = answered ? "Grounded answer" : "Insufficient evidence";
  pill.className = "pill " + (answered ? "ok" : "warn");
  $("answer-meta").textContent =
    `Read tool · ${d.model} · ${d.index} index · top ${d.top_k} · retrieval ${d.timings_ms.retrieval} ms · ` +
    `generation ${d.timings_ms.generation} ms` + (d.reason && d.reason !== "model abstained" ? ` · ${d.reason}` : "");

  const list = $("sources");
  list.replaceChildren();
  for (const s of data.sources) {
    const li = el("li");
    li.append(el("div", "src-title", `[${s.rank}] ${s.section || "(no section)"}`));
    li.append(el("div", "src-meta",
      `${s.document_title || s.document} · ${pages(s.pages)} · similarity ${s.similarity_score.toFixed(4)}`));
    li.append(el("div", "chunk-id", s.chunk_id));
    if (s.quote) li.append(el("p", "src-quote", `“${s.quote}”`));
    list.append(li);
  }
  $("no-sources").hidden = data.sources.length > 0;

  const chunks = $("chunks");
  chunks.replaceChildren();
  for (const c of d.retrieved) {
    const li = el("li");
    const head = el("div", "chunk-head");
    head.append(el("span", "score", `#${c.rank} · ${c.similarity_score.toFixed(4)}`));
    head.append(el("span", "", `${c.section || "(no section)"} · ${pages(c.pages)}`));
    head.append(el("span", "tag" + (c.cited ? "" : " no"), c.cited ? "cited" : "not cited"));
    li.append(head, el("div", "chunk-id", c.chunk_id), el("pre", "chunk-text", c.text));
    chunks.append(li);
  }
  const rejected = d.rejected_citations.length
    ? ` Rejected citations: ${d.rejected_citations.map((r) => `${r.chunk_id} (${r.reason})`).join("; ")}.` : "";
  $("debug-note").textContent =
    `Similarity scores rank chunks; they do not prove an answer exists.${rejected}`;
  $("result").hidden = false;
}

// ---- ask -------------------------------------------------------------------
async function ask(question) {
  if (pendingAction) {
    showError("Resolve the pending request", "Choose Confirm simulation or Cancel before sending another message.");
    return;
  }
  $("error").hidden = true;
  $("action-panel").hidden = true;
  $("result").hidden = true;
  $("loading").hidden = false;
  $("ask-btn").disabled = true;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      signal: controller.signal,
    });
    const data = await res.json().catch(() => null);
    if (!res.ok || !data || data.error) {
      const err = (data && data.error) || { code: `http_${res.status}`, message: "The server returned an unexpected response." };
      const titles = {
        ollama_unavailable: "Ollama is not running",
        model_missing: "Model not installed",
        ollama_timeout: "The local model timed out",
        chunks_missing: "Document chunks are missing",
        retrieval_failed: "Retrieval failed",
        malformed_output: "The model returned malformed output",
        bad_request: "Check your question",
        invalid_input: "Check your leave request",
      };
      showError(titles[err.code] || "Something went wrong", err.message);
      return;
    }
    if (data.action) renderAction(data.action);
    else render(data);
  } catch (e) {
    showError(e.name === "AbortError" ? "The request timed out" : "Cannot reach the app server",
      e.name === "AbortError" ? "No answer within the time limit. Try again; the first question loads the model."
        : "Is 'python run_rag.py web' still running?");
  } finally {
    clearTimeout(timer);
    $("loading").hidden = true;
    $("ask-btn").disabled = !!pendingAction;
  }
}

$("ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("question").value.trim();
  if (q) ask(q);
});
$("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) $("ask-form").requestSubmit();
});
loadHealth();
fetch("/api/action/pending").then(r => r.json()).then(data => {
  if (data.action) renderAction(data.action);
}).catch(() => {});
