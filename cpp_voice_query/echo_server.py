#!/usr/bin/env python3
"""
Minimal echo server that mimics llama.cpp's /v1/chat/completions SSE API.
Echoes the last user message back word-by-word as a streaming response.

Usage: python3 echo_server.py [port]   (default: 8001)
"""

import json
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler


class EchoHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        # Extract last user message
        messages = body.get("messages", [])
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break

        echo = f"{user_msg}"
        words = echo.split()

        # Stream as SSE, word by word
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            chunk = {
                "choices": [{"delta": {"content": token}, "index": 0}]
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.05)

        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, fmt, *args):
        print(f"[echo] {args[0]}")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    server = HTTPServer(("0.0.0.0", port), EchoHandler)
    print(f"Echo server listening on http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
