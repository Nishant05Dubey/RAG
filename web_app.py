"""Day 07 local web app: one page plus a small JSON API over qa_pipeline.

    python run_rag.py web                 # http://127.0.0.1:5000
    python web_app.py --port 8000

Endpoints:
    GET  /             the RAG page (templates/index.html)
    GET  /api/health   can the app read the chunks and reach Ollama + the model?
    POST /api/ask      {"question": "..."} -> {answer, citations, sources, debug}

Errors come back as {"error": {"code", "message"}} with a matching HTTP status.
Every question asked in the browser is appended to logs/web_qa_log.jsonl.
"""
import argparse
import json
import sys
import secrets
from datetime import datetime

from flask import Flask, jsonify, render_template, request, session

from paths import LOGS_DIR
from qa_pipeline import MAX_QUESTION_CHARS, QAError, QAPipeline
from day10_tools import LocalTools, TOOL_LOG, ToolError, is_leave_request

WEB_LOG = LOGS_DIR / "web_qa_log.jsonl"


def log_answer(question, result=None, error=None):
    rec = {"timestamp": datetime.now().isoformat(timespec="seconds"), "question": question}
    if result:
        d = result["debug"]
        rec.update(status=d["status"], answer=result["answer"], citations=result["citations"],
                   model=d["model"], index=d["index"], timings_ms=d["timings_ms"],
                   retrieved=[(r["chunk_id"], r["similarity_score"]) for r in d["retrieved"]])
    if error:
        rec["error"] = error
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with WEB_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def error_response(code, message, status, debug=None):
    body = {"error": {"code": code, "message": message}}
    if debug:
        body["debug"] = debug
    return jsonify(body), status


def create_app(pipeline=None, log=True, tool_log_path=TOOL_LOG):
    app = Flask(__name__)
    app.secret_key = secrets.token_hex(32)
    app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Strict')
    app.json.sort_keys = False  # keep answer, citations, sources, debug in order
    pipeline = pipeline or QAPipeline()
    tools = LocalTools(pipeline, tool_log_path if log else None)
    app.extensions['day10_tools'] = tools

    def owner():
        if 'tool_owner' not in session:
            session['tool_owner'] = secrets.token_urlsafe(24)
        return session['tool_owner']

    @app.errorhandler(ToolError)
    def tool_error(error):
        return error_response(error.code, str(error), error.http_status)

    @app.get('/api/action/pending')
    def pending_action():
        return jsonify(action=tools.pending(owner()))

    @app.post('/api/action/decision')
    def decide_action():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            raise ToolError('Expected a confirmation or cancellation.')
        try:
            return jsonify(action=tools.decide(body.get('action_id'), body.get('decision'), owner()))
        except ToolError as exc:
            tools.log('simulate_leave', 'rejected', simulated=True, error=str(exc))
            raise
        except OSError:
            return error_response('log_unavailable', 'Could not save the simulation log. No action was executed; try again.', 503)

    @app.get("/")
    def index():
        return render_template("index.html", max_chars=MAX_QUESTION_CHARS)

    @app.get("/api/health")
    def health():
        h = pipeline.health()
        return jsonify(h), 200 if h["ok"] else 503

    @app.post("/api/ask")
    def ask():
        body = request.get_json(silent=True) or {}
        if not isinstance(body, dict):
            return error_response('bad_request', 'Please send a question.', 400)
        question = body.get("question")
        if not isinstance(question, str) or not question.strip():
            return error_response("bad_request", "Please type a question.", 400)
        if len(question) > MAX_QUESTION_CHARS:
            return error_response("bad_request",
                                  f"Questions are limited to {MAX_QUESTION_CHARS} characters.", 400)
        try:
            if is_leave_request(question):
                return jsonify(action=tools.stage_leave(question, owner()))
            result = tools.read_policy(question)
        except ToolError:
            raise
        except QAError as e:
            if log:
                log_answer(question, error=e.code)
            return error_response(e.code, str(e), e.http_status, e.debug)
        except Exception:
            app.logger.exception("Unexpected error while answering")
            if log:
                log_answer(question, error="internal_error")
            return error_response("internal_error",
                                  "Unexpected server error; see the terminal running the app.", 500)
        if log:
            log_answer(question, result)
        return jsonify(result)

    return app


def main(argv=None):
    p = argparse.ArgumentParser(description="Run the Blend RAG Assistant locally.")
    p.add_argument("--host", default="127.0.0.1", help="default 127.0.0.1 (this computer only)")
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args(argv)

    pipeline = QAPipeline()
    print("Loading the retriever (embedding model + chunks)...")
    try:
        r = pipeline.retriever
        print(f"  {len(r.chunks)} chunks ({pipeline.index} index) from {r.document_metadata.get('document_title')}")
    except QAError as e:
        print(f"  WARNING: {e}")
    h = pipeline.health()["ollama"]
    print(f"Ollama model '{h['model']}': " + ("ready" if h["ok"] else f"NOT READY - {h.get('message')}"))
    print(f"\nOpen http://{args.host}:{args.port} in your browser (Ctrl+C to stop).\n")
    create_app(pipeline).run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
