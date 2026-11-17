"""Night dual-instance least-loaded router: two loopback backends, in-flight
counting, no queueing beyond the backends' own capacity."""
import argparse
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Backends:
    def __init__(self, urls):
        self.urls = list(urls)
        self.inflight = [0] * len(urls)
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            index = min(range(len(self.urls)), key=lambda i: self.inflight[i])
            self.inflight[index] += 1
            return index

    def release(self, index):
        with self.lock:
            self.inflight[index] -= 1


def make_handler(backends):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _forward(self, method):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else None
            index = backends.acquire()
            try:
                req = urllib.request.Request(
                    backends.urls[index] + self.path,
                    data=body,
                    method=method,
                    headers={"Content-Type": self.headers.get("Content-Type", "application/json")},
                )
                try:
                    with urllib.request.urlopen(req, timeout=3600) as r:
                        payload, status = r.read(), r.status
                except urllib.error.HTTPError as exc:
                    payload, status = exc.read(), exc.code
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            finally:
                backends.release(index)

        def do_GET(self):
            # Health passthrough: any backend healthy means the router serves.
            for url in backends.urls:
                try:
                    with urllib.request.urlopen(url + self.path, timeout=10) as r:
                        payload, status = r.read(), r.status
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                except Exception:
                    continue
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):
            self._forward("POST")

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--backend", action="append", required=True)
    args = parser.parse_args()
    backends = Backends(args.backend)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(backends))
    print(json.dumps({"router": args.port, "backends": args.backend}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
