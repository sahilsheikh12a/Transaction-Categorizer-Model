"""A local web UI for trying the categorizer by hand.

    python -m phonepe_categorizer serve

Binds to 127.0.0.1 only. This is financial data and this server has no
authentication, so it must never listen on a public interface — `--host` exists
for cases like a container port-forward, and warns when you use it.

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
from .export import OUTPUT_COLUMNS
from .importer import UnresolvedHeaders, parse
from .normalize import normalize_merchant

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 32 * 1024 * 1024  # 32 MB — well past any real statement


class _State:
    """Server-lifetime handle on the learned state."""

    def __init__(self, db_path: str | None):
        self.db_path = db_path

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
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_UPLOAD:
            raise ValueError(f"upload exceeds {MAX_UPLOAD // (1024 * 1024)} MB")
        return self.rfile.read(length) if length else b""

    # ── routes ──────────────────────────────────────────────────────────
    def do_GET(self) -> None:  # noqa: N802
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
        route = urlparse(self.path).path
        try:
            if route == "/api/classify":
                self._classify()
            elif route == "/api/categorize":
                self._categorize()
            elif route == "/api/merchants":
                self._merchants()
            elif route == "/api/correct":
                self._correct()
            else:
                self._json({"error": "not found"}, 404)
        except UnresolvedHeaders as exc:
            self._json({"error": str(exc), "missing": exc.missing,
                        "headers": exc.headers}, 422)
        except Exception as exc:  # keep the server alive on any handler bug
            traceback.print_exc()
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def _classify(self) -> None:
        payload = json.loads(self._body() or b"{}")
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

        spellings: dict[str, dict[str, int]] = {}
        for t in result.rows:
            key = normalize_merchant(t.counterparty) or t.counterparty
            spellings.setdefault(key, {})
            spellings[key][t.counterparty] = spellings[key].get(t.counterparty, 0) + 1

        out = []
        for variants in spellings.values():
            name = max(variants.items(), key=lambda kv: (kv[1], kv[0]))[0]
            # Outgoing on purpose — see export.merchant_map_csv.
            res = classify_core(name, is_credit=False, direction="Paid to",
                                overrides=overrides, learned=learned)
            out.append({"merchant": name, "category": res.category,
                        "needsReview": res.needs_review})
        out.sort(key=lambda r: r["merchant"].lower())
        self._json({"merchants": out})

    def _correct(self) -> None:
        payload = json.loads(self._body() or b"{}")
        merchant = (payload.get("merchant") or "").strip()
        category = (payload.get("category") or "").strip()
        if not merchant or category not in C.VALID:
            self._json({"error": "merchant and a valid category are required"}, 400)
            return
        self._json(STATE.correct(merchant, category))


def serve(host: str = "127.0.0.1", port: int = 8000, db: str | None = "categorizer.db") -> int:
    STATE.db_path = db
    if host not in ("127.0.0.1", "localhost", "::1"):
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
