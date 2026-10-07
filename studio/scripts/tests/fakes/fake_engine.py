#!/usr/bin/env python3
"""A stand-in for the oMLX HTTP surface the update script talks to.

    fake_engine.py omlx|ds4 <engines home> <port>

oMLX answers /api/status, /v1/models and POST /v1/models/<id>/unload. It reports 500 on
/v1/models while <home>/omlx/VERSION says "bad", which is how a test makes a venv unhealthy.
It models the graceful-unload contract: while <home>/omlx-unload-busy exists a plain unload answers
409 {"error": {"type": "model_busy"}} with Retry-After 15 and unloads nothing; "?force=1" aborts and
unloads (200). The calls log gets omlx-unload, omlx-unload-busy or omlx-unload-force.
ds4 stands for the legacy DwarfStar launcher of a pre-removal app (migration tests only): it answers
/admin/status and POST /admin/stop; <home>/ds4-busy makes it report a request in flight and
<home>/ds4-starting that it is starting.

"""

import json
import os
import signal
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

# A harness that starts the tests can leave SIGTERM/SIGINT/SIGHUP ignored, and children inherit that:
# a fake that cannot be terminated would outlive the run. Restore the defaults, and end by itself
# after 15 minutes whatever happens.
for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(_sig, signal.SIG_DFL)
signal.alarm(900)

kind, home, port = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
CALLS = Path(os.environ["FAKE_LC"]) / "calls.log"


def flag(path: Path) -> bool:
    return path.exists()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, code, body=None, headers=None):
        data = json.dumps(body if body is not None else {}).encode()
        self.send_response(code)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if kind == "omlx":
            version = (home / "omlx" / "VERSION").read_text().strip() if (home / "omlx" / "VERSION").exists() else ""
            if self.path == "/api/status":
                loaded = ["m1"] if flag(home / "omlx-loaded") else []
                return self.reply(200, {"active_requests": 1 if flag(home / "omlx-busy") else 0, "waiting_requests": 0, "models_loading": 0, "loaded_models": loaded})
            if self.path == "/v1/models":
                return self.reply(500 if version == "bad" else 200, {"data": [{"id": "m1"}, {"id": "m2"}]})
        elif self.path == "/admin/status":
            return self.reply(
                200,
                {
                    "loaded": flag(home / "ds4-loaded"),
                    "in_flight": 1 if flag(home / "ds4-busy") else 0,
                    "starting": flag(home / "ds4-starting"),
                },
            )
        self.reply(404)

    def do_POST(self):
        url = urlsplit(self.path)
        if kind == "omlx" and url.path.startswith("/v1/models/") and url.path.endswith("/unload"):
            force = parse_qs(url.query).get("force") == ["1"]
            if flag(home / "omlx-unload-busy") and not force:
                CALLS.open("a").write("omlx-unload-busy\n")
                return self.reply(
                    409,
                    {"error": {"type": "model_busy", "message": "model is serving a request"}},
                    {"Retry-After": "15"},
                )
            (home / "omlx-loaded").unlink(missing_ok=True)
            CALLS.open("a").write("omlx-unload-force\n" if force else "omlx-unload\n")
            return self.reply(200)
        if kind == "ds4" and self.path.startswith("/admin/stop"):
            (home / "ds4-loaded").unlink(missing_ok=True)
            CALLS.open("a").write(f"ds4-stop {self.path}\n")
            return self.reply(200)
        self.reply(404)


HTTPServer(("127.0.0.1", port), Handler).serve_forever()
