"""No failed/partial request may qualify a throughput result."""

import importlib.util
import json
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from contextlib import contextmanager
import urllib.request
import pytest


def load(name):
    s = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "tools" / (name + ".py")
    )
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


blast = load("night_blast")
router = load("night_router")


@contextmanager
def backend(kind="ok"):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            count = body["n_predict"]
            obj = {"tokens": [1] * (count if kind == "ok" else count - 1)}
            data = json.dumps(obj).encode()
            self.send_response(503 if kind == "http-error" else 200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    class Server(ThreadingHTTPServer):
        request_queue_size = 128

    s = Server(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=s.serve_forever, daemon=True)
    t.start()
    try:
        yield s.server_address[1]
    finally:
        s.shutdown()
        s.server_close()
        t.join()


@pytest.mark.parametrize("kind", ["ok", "short", "http-error"])
def test_complete_only_tps(kind):
    with backend(kind) as port:
        r = blast.blast(port, 4, 3, 2, 32, 5)
    assert r["valid"] == (kind == "ok")
    assert (r["aggregate_tps"] is not None) == r["valid"]
    assert len(r["errors"]) == (0 if kind == "ok" else 4)
    assert r["expected_tokens"] == 12


def test_router_balances_and_completes_c32():
    with backend() as p1, backend() as p2:
        b = router.Backends([f"http://127.0.0.1:{p1}", f"http://127.0.0.1:{p2}"])
        s = router.BenchmarkRouter(("127.0.0.1", 0), router.make_handler(b))
        t = threading.Thread(target=s.serve_forever, daemon=True)
        t.start()
        try:
            r = blast.blast(s.server_address[1], 32, 3, 2, 32, 10)
        finally:
            s.shutdown()
            s.server_close()
            t.join()
        assert r["valid"] and r["emitted_tokens"] == 96 and not r["errors"]
        assert b.inflight == [0, 0]


def test_tie_routing_rotates_and_release_is_balanced():
    b = router.Backends(["a", "b"])
    chosen = []
    for _ in range(4):
        i = b.acquire()
        chosen.append(i)
        b.release(i)
    assert chosen == [0, 1, 0, 1]


def test_timing_output_requires_exact_qualified_input_and_budget(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "tools"))
    from night_blast import qualified_output_ids
    import copy

    report = dict(
        completed=True,
        passed=True,
        implementations={
            "dequant": {
                "cases": [
                    dict(name="semantic", generated_ids=[9] * 8),
                    dict(
                        name="timing",
                        prompt_token_ids=[1, 2],
                        max_new_tokens=3,
                        generated_ids=[4, 5, 6],
                    ),
                ]
            }
        },
    )
    assert qualified_output_ids(report, [1, 2], 3) == [4, 5, 6]
    for prompt, count in (([2, 1], 3), ([1, 2], 2), ([1, 2], 32)):
        with pytest.raises(ValueError):
            qualified_output_ids(report, prompt, count)
    legacy = copy.deepcopy(report)
    legacy["implementations"]["dequant"]["cases"].pop()
    with pytest.raises(ValueError):
        qualified_output_ids(legacy, [1, 2], 3)
    report["implementations"]["dequant"]["cases"][-1]["generated_ids"].pop()
    with pytest.raises(ValueError):
        qualified_output_ids(report, [1, 2], 3)


@pytest.mark.parametrize(
    "dims",
    [
        (0, 32, 16, 100),
        (1, 0, 16, 100),
        (1, 32, 0, 100),
        (1, 32, 16, 0),
        (True, 32, 16, 100),
    ],
)
def test_blast_rejects_dimensions_before_building_prompt(monkeypatch, dims):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "tools"))
    from night_blast import blast

    with pytest.raises(ValueError):
        blast(1, *dims, 1)
