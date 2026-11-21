"""Transport and receipt contracts for the pinned vLLM OpenAI worker."""

from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest

from failslow.engines import cache_key
from failslow.native_server import NativeServer, streamed_tokens


def chunk(tag, ids, finish=None):
    data = {"id": tag, "choices": [{"token_ids": ids, "finish_reason": finish}]}
    return b"data: " + json.dumps(data).encode() + b"\n\n"


class NativeServing(unittest.TestCase):
    def test_rejects_grouped_tokens_and_missing_terminal_event(self):
        with self.assertRaisesRegex(RuntimeError, "Multiple token"):
            list(
                streamed_tokens(
                    io.BytesIO(chunk("x", [2, 3], "length") + b"data: [DONE]\n"), "x"
                )
            )
        with self.assertRaisesRegex(RuntimeError, "before"):
            list(streamed_tokens(io.BytesIO(chunk("x", [1], "length")), "x"))
        with self.assertRaisesRegex(RuntimeError, "finish reason"):
            list(streamed_tokens(io.BytesIO(chunk("x", [1]) + b"data: [DONE]\n"), "x"))

    def exercise(self, failure=None):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            client = NativeServer(root, 0, root / "log")
            ids = list(range(65))
            folder = root / cache_key(ids)
            folder.mkdir()

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_):
                    pass

                def do_POST(self):
                    body = json.loads(
                        self.rfile.read(int(self.headers["Content-Length"]))
                    )
                    if (
                        self.path != "/v1/completions"
                        or not body["stream"]
                        or not body["return_token_ids"]
                    ):
                        self.send_error(400)
                        return
                    key = body["request_id"]
                    tag = "cmpl-" + key
                    if self.headers["X-Request-Id"] != key:
                        self.send_error(400)
                        return
                    if failure != "missing":
                        (root / "forward.jsonl").write_text(
                            "".join(
                                json.dumps(
                                    [
                                        key if failure != "wrong" else "other",
                                        t,
                                        t + 0.1,
                                        False,
                                    ]
                                )
                                + "\n"
                                for t in [1, 2]
                            )
                        )
                    (folder / "loaded.json").write_text(
                        json.dumps(
                            {"loaded_tokens": 16 if failure == "receipt" else 64}
                        )
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    self.wfile.write(
                        chunk(tag, [4])
                        + chunk(tag, [9], "length")
                        + b"data: [DONE]\n\n"
                    )

            with HTTPServer(("127.0.0.1", 0), Handler) as http:
                client.url = f"http://127.0.0.1:{http.server_port}"
                thread = threading.Thread(target=http.handle_request)
                thread.start()
                rows = []
                try:
                    proof = client.request(
                        dict(
                            mode="decode",
                            original=ids,
                            ids=ids + [20],
                            count=2,
                            origin=0,
                            interval=(0, 0),
                            factor=1.0,
                        ),
                        lambda *a: rows.append(a),
                    )
                finally:
                    thread.join(4)
                    self.assertFalse((root / "active.json").exists())
                self.assertEqual(proof["loaded_tokens"], 64)
                self.assertEqual([r[0] for r in rows], [4, 9])
                self.assertEqual([r[2] for r in rows], [1, 2])

    def test_request_ids_and_real_loopback_stream(self):
        self.exercise()

    def test_unjoined_or_missing_forward_fails(self):
        for fault in ["wrong", "missing", "receipt"]:
            with self.subTest(fault=fault), self.assertRaises(RuntimeError):
                self.exercise(fault)
