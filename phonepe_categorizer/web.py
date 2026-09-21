"""A local web UI for trying the categorizer by hand.

    python -m phonepe_categorizer serve

Binds to 127.0.0.1 only. This is financial data and this server has no
authentication, so it must never listen on a public interface — `--host` exists
for cases like a container port-forward, and warns when you use it.

Listening on localhost is not enough on its own: any web page open in the same
browser can send requests to it. Two checks close that:

  * POSTs must carry the content type their endpoint expects. JSON and CSV are
    not "simple" types, so a cross-site page must ask the browser's permission
    first (a CORS preflight), and this server never grants it. Without this, a
    page could POST text/plain and quietly write corrections into the DB.
  * While bound to loopback, the Host header must be a loopback name. That
    defeats DNS rebinding, where a hostile domain re-resolves to 127.0.0.1 to
    make its requests look same-origin.

Built on `http.server` rather than a framework on purpose: this is a testing
tool, and a zero-dependency one can be run straight from a clean checkout. The
handlers are thin wrappers over the same functions the CLI calls, so what you
see in the browser is what the library actually does.
"""
from __future__ import annotations

import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from . import categories as C
from .db import CategorizerRepository, open_session
from .engine import classify_core
from .export import OUTPUT_COLUMNS, merchant_map
from .importer import UnresolvedHeaders, parse

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 32 * 1024 * 1024  # 32 MB — well past any real statement
LOOPBACK = ("127.0.0.1", "localhost", "::1")

# The content type each POST endpoint accepts. See the module docstring.
JSON, CSV = "application/json", "text/csv"
POST_TYPES = {
    "/api/classify": JSON,
    "/api/correct": JSON,
    "/api/categorize": CSV,
    "/api/merchants": CSV,
}


class BadRequest(ValueError):
    """The client sent something unusable. Answered with 400, not 500."""


class _State:
    """Server-lifetime handle on the learned state."""

    def __init__(self, db_path: str | None):
        self.db_path = db_path
        self.loopback_only = True

    def _session(self):
        return open_session(f"sqlite:///{self.db_path}")

    def learned(self) -> tuple[dict[str, str], dict[str, str]]:
        if not self.db_path:
            return {}, {}
        with self._session() as s:
            repo = CategorizerRepository(s)
            return repo.overrides(), repo.learned_merchants()

    def correct(self, merchant: str, category: str) -> dict:
        if not self.db_path:
            return {"updated": 0, "persisted": False}
        with self._session() as s:
            out = CategorizerRepository(s).record_correction(merchant, category)
        return {**out, "persisted": True}


STATE = _State(None)


def _classify_rows(rows, overrides, learned):
    """Classify parsed rows, memoized by merchant + direction."""
    memo: dict[tuple, object] = {}
    out = []
    for t in rows:
        key = (t.counterparty, t.is_credit, t.direction)
        res = memo.get(key)
        if res is None:
            res = classify_core(
                t.counterparty, is_credit=t.is_credit, direction=t.direction,
                overrides=overrides, learned=learned,
            )
            memo[key] = res
        out.append({
            "date": t.date.strftime("%Y-%m-%d"),
            "counterparty": t.counterparty,
            "direction": t.direction,
            "amount": t.amount,
            "isCredit": t.is_credit,
            **res.as_dict(),
        })
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "phonepe-categorizer"

    # Quieten the default one-line-per-request logging.
    def log_message(self, fmt, *args):  # noqa: A003
        pass

    # ── helpers ─────────────────────────────────────────────────────────
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json; charset=utf-8")

    def _body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise BadRequest("invalid Content-Length") from None
        if length < 0:
            raise BadRequest("invalid Content-Length")
        if length > MAX_UPLOAD:
            raise BadRequest(f"upload exceeds {MAX_UPLOAD // (1024 * 1024)} MB")
        return self.rfile.read(length) if length else b""

    def _json_body(self) -> dict:
        try:
            payload = json.loads(self._body() or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise BadRequest("body is not valid JSON") from None
        if not isinstance(payload, dict):
            raise BadRequest("body must be a JSON object")
        return payload

    def _host_allowed(self) -> bool:
        """While bound to loopback, only answer to loopback host names."""
        if not STATE.loopback_only:
            return True
        host = (self.headers.get("Host") or "").strip().lower()
        if host.startswith("["):                      # [::1]:8000
            name = host[1:host.find("]")] if "]" in host else host
        else:
            name = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
        return name in LOOPBACK

    # ── routes ──────────────────────────────────────────────────────────
    def do_GET(self) -> None:  # noqa: N802
        if not self._host_allowed():
            self._json({"error": "forbidden host"}, 403)
            return
        route = urlparse(self.path).path
        if route in ("/", "/index.html"):
            page = STATIC / "index.html"
            if not page.exists():
                self._send(500, b"index.html missing", "text/plain")
                return
            self._send(200, page.read_bytes(), "text/html; charset=utf-8")
        elif route == "/favicon.ico":
            # Browsers request this unprompted; answering 204 keeps a spurious
            # 404 out of the console when someone is debugging the page.
            self.send_response(204)
            self.end_headers()
        elif route == "/api/categories":
            self._json({
                "categories": list(C.ALL),
                "definitions": C.DEFINITIONS,
                "sources": list(C.SOURCES),
            })
        elif route == "/api/state":
            overrides, learned = STATE.learned()
            self._json({
                "db": STATE.db_path,
                "overrides": len(overrides),
                "learnedMerchants": len(learned),
            })
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        if not self._host_allowed():
            self._json({"error": "forbidden host"}, 403)
            return
        route = urlparse(self.path).path
        expected = POST_TYPES.get(route)
        if expected is None:
            self._json({"error": "not found"}, 404)
            return
        # An empty body cannot change anything (every endpoint rejects it), so
        # only a request that carries data has to prove its content type.
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if self.headers.get("Content-Length", "0") != "0" and ctype != expected:
            self._json({"error": f"Content-Type must be {expected}"}, 415)
            return
        try:
            if route == "/api/classify":
                self._classify()
            elif route == "/api/categorize":
                self._categorize()
            elif route == "/api/merchants":
                self._merchants()
            else:
                self._correct()
        except BadRequest as exc:
            self._json({"error": str(exc)}, 400)
        except UnresolvedHeaders as exc:
            self._json({"error": str(exc), "missing": exc.missing,
                        "headers": exc.headers}, 422)
        except Exception as exc:  # keep the server alive on any handler bug
            traceback.print_exc()
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def _classify(self) -> None:
        payload = self._json_body()
        text = (payload.get("text") or "").strip()
        if not text:
            self._json({"error": "text is required"}, 400)
            return
        overrides, learned = STATE.learned()
        res = classify_core(
            text,
            is_credit=bool(payload.get("isCredit")),
            direction=payload.get("direction") or "",
            overrides=overrides, learned=learned,
        )
        self._json({"input": text, **res.as_dict()})

    def _categorize(self) -> None:
        raw = self._body()
        if not raw:
            self._json({"error": "empty upload"}, 400)
            return
        result = parse(raw, dedupe=False)
        overrides, learned = STATE.learned()
        rows = _classify_rows(result.rows, overrides, learned)

        by_category: dict[str, int] = {}
        for r in rows:
            by_category[r["category"]] = by_category.get(r["category"], 0) + 1
        by_source: dict[str, int] = {}
        for r in rows:
            by_source[r["source"]] = by_source.get(r["source"], 0) + 1

        self._json({
            "summary": {
                "inputRows": result.total_data_rows,
                "classified": len(rows),
                "needsReview": sum(1 for r in rows if r["needsReview"]),
                "other": sum(1 for r in rows if r["category"] == C.OTHER),
                "skipped": result.skipped,
                "problems": result.problem_counts(),
                "uniqueMerchants": len({r["normalizedMerchant"] for r in rows}),
            },
            "byCategory": by_category,
            "bySource": by_source,
            "columns": list(OUTPUT_COLUMNS),
            "rows": rows,
        })

    def _merchants(self) -> None:
        """Unique merchant -> category, for the two-column download."""
        raw = self._body()
        if not raw:
            self._json({"error": "empty upload"}, 400)
            return
        result = parse(raw, dedupe=False)
        overrides, learned = STATE.learned()
        self._json({"merchants": [
            {"merchant": name, "category": category, "needsReview": review}
            for name, category, review in merchant_map(result.rows, overrides, learned)
        ]})

    def _correct(self) -> None:
        payload = self._json_body()
        merchant = (payload.get("merchant") or "").strip()
        category = (payload.get("category") or "").strip()
        if not merchant or category not in C.VALID:
            self._json({"error": "merchant and a valid category are required"}, 400)
            return
        self._json(STATE.correct(merchant, category))


def serve(host: str = "127.0.0.1", port: int = 8000, db: str | None = "categorizer.db") -> int:
    STATE.db_path = db
    STATE.loopback_only = host in LOOPBACK
    if not STATE.loopback_only:
        print(f"WARNING: binding to {host}. This server has no authentication and "
              f"handles financial data — do not expose it to an untrusted network.")
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"phonepe-categorizer UI  →  http://{host}:{port}")
    print(f"learned state: {db or '(none — corrections will not persist)'}")
    print("Ctrl-C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0


__all__ = ["serve", "Handler"]
