#!/usr/bin/env python3
"""A stand-in for the oMLX HTTP surface the update script talks to.

    fake_engine.py omlx <engines home> <port>

oMLX answers /api/status, /v1/models and POST /v1/models/<id>/unload. It reports 500 on
/v1/models while <home>/omlx/VERSION says "bad", which is how a test makes a venv unhealthy.

"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

kind, home, port = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
CALLS = Path(os.environ["FAKE_LC"]) / "calls.log"


def flag(path: Path) -> bool:
    return path.exists()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, code, body=None):
        data = json.dumps(body if body is not None else {}).encode()
        self.send_response(code)
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
        self.reply(404)

    def do_POST(self):
        if kind == "omlx" and self.path.startswith("/v1/models/") and self.path.endswith("/unload"):
            (home / "omlx-loaded").unlink(missing_ok=True)
            CALLS.open("a").write("omlx-unload\n")
            return self.reply(200)
        self.reply(404)


HTTPServer(("127.0.0.1", port), Handler).serve_forever()
