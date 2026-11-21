"""Native vLLM OpenAI server and its request-correlated single-sequence adapter."""

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, build_opener, ProxyHandler
import uuid

from .data import write_json
from .engines import BLOCK, MODEL, REVISION, cache_key
from .text_workload import TextWorkload


def streamed_tokens(response, completion_id):
    finished = False
    for raw in response:
        line = raw.decode("utf-8").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            raise RuntimeError("Malformed completion SSE")
        data = line[5:].strip()
        if data == "[DONE]":
            if not finished:
                raise RuntimeError("Missing completion finish reason")
            return
        msg = json.loads(data)
        if "error" in msg or msg.get("id") != completion_id:
            raise RuntimeError(f"vLLM request failed or mismatched ID: {msg}")
        for choice in msg["choices"]:
            ids = choice.get("token_ids") or []
            if len(ids) > 1:
                raise RuntimeError(
                    "Multiple token IDs in one chunk; exact client ITL unavailable"
                )
            if ids:
                yield int(ids[0])
            finished |= choice.get("finish_reason") is not None
    raise RuntimeError("Completion stream ended before [DONE]")


class NativeServer:
    def __init__(self, root, device, log_path):
        self.root = Path(root)
        self.device = str(device)
        self.log_path = Path(log_path)
        self.proc = None
        self.log = None
        self.http = build_opener(ProxyHandler({}))
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"

    def start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("a")
        config = dict(
            kv_connector="ServerTimingConnector",
            kv_connector_module_path="failslow.connector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": str(self.root)},
        )
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": self.device,
            "VLLM_USE_V1": "1",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
            "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
            "VLLM_KV_CACHE_LAYOUT": "NHD",
            "TOKENIZERS_PARALLELISM": "false",
        }
        args = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--model",
            MODEL,
            "--revision",
            REVISION,
            "--tokenizer-revision",
            REVISION,
            "--dtype",
            "float16",
            "--generation-config",
            "vllm",
            "--enforce-eager",
            "--max-model-len",
            "512",
            "--block-size",
            str(BLOCK),
            "--max-num-batched-tokens",
            "512",
            "--max-num-seqs",
            "1",
            "--no-enable-prefix-caching",
            "--distributed-executor-backend",
            "uni",
            "--gpu-memory-utilization",
            "0.65",
            "--kv-transfer-config",
            json.dumps(config),
        ]
        self.proc = subprocess.Popen(
            args,
            env=env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def ready(self, deadline):
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vLLM exited during startup. See {self.log_path}")
            try:
                with self.http.open(self.url + "/v1/models", timeout=3) as resp:
                    models = json.load(resp)["data"]
                if MODEL not in [m["id"] for m in models]:
                    raise RuntimeError(f"Unexpected model on {self.url}")
                with self.http.open(self.url + "/version", timeout=3) as resp:
                    version = json.load(resp)["version"]
                if version.split("+")[0] != "0.10.2":
                    raise RuntimeError(f"Unsupported vLLM server {version}")
                return version
            except (URLError, TimeoutError, ConnectionError):
                time.sleep(1)
        raise TimeoutError(f"vLLM startup timed out. See {self.log_path}")

    def close(self):
        if self.proc is not None:
            # Children can survive a failed frontend. Shut down only the owned group.
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=12)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait()
        if self.log:
            self.log.close()

    def request(self, job, on_token):
        key = uuid.uuid4().hex
        count = job["count"]
        mode = job["mode"]
        folder = self.root / cache_key(job["original"])
        receipt = folder / "loaded.json"
        if (mode == "decode" and (not folder.is_dir() or receipt.exists())) or (
            mode != "decode" and folder.exists()
        ):
            raise RuntimeError("Unexpected KV state before native vLLM request")
        control = self.root / "active.json"
        spans = self.root / "forward.jsonl"
        if control.exists():
            raise RuntimeError(
                "Concurrent request on a serialized instrumentation slot"
            )
        spans.write_text("")
        write_json(
            control,
            {
                "request_key": key,
                **{k: job[k] for k in ("origin", "interval", "factor")},
            },
        )
        body = dict(
            model=MODEL,
            prompt=job["ids"],
            temperature=0,
            max_tokens=count,
            ignore_eos=True,
            add_special_tokens=False,
            request_id=key,
            stream=True,
            return_token_ids=True,
        )
        req = Request(
            self.url + "/v1/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "X-Request-Id": key},
        )
        num = 0
        try:
            with self.http.open(req, timeout=180) as resp, spans.open() as records:
                for tok in streamed_tokens(resp, "cmpl-" + key):
                    # The worker closes this record before the server samples and emits a token.
                    line = records.readline()
                    if not line:
                        raise RuntimeError(
                            "No worker forward timing for streamed token"
                        )
                    tag, start, end, applied = json.loads(line)
                    if tag != key or end <= start or num >= count:
                        raise RuntimeError(
                            "Invalid or uncorrelated worker forward timing"
                        )
                    on_token(tok, num, start, end, applied)
                    num += 1
                if num != count or records.readline():
                    raise RuntimeError("Forward count and streamed token count differ")
            proof = {}
            if mode == "decode":
                proof = json.loads(receipt.read_text())
                expected = (len(job["original"]) - 1) // BLOCK * BLOCK
                if expected < 1 or proof.get("loaded_tokens") != expected:
                    raise RuntimeError(
                        "Decode did not import the expected external KV prefix"
                    )
            return proof
        finally:
            import shutil

            control.unlink(missing_ok=True)
            if mode in ("decode", "reference") and folder.is_dir():
                shutil.rmtree(folder)


def boot_servers(cluster, log_dir):
    import torch
    from importlib.metadata import version

    if version("vllm").split("+")[0] != "0.10.2" or not torch.cuda.is_available():
        raise RuntimeError(
            "The native split-server collector requires vLLM 0.10.2 and CUDA"
        )
    cluster.text = TextWorkload()
    ids = cluster.topology["prefill"] + cluster.topology["decode"]
    end = time.monotonic() + 1200
    for i, (name, device) in enumerate(zip(ids, cluster.devices)):
        prop = torch.cuda.get_device_properties(i)
        if prop.major < 8:
            raise RuntimeError(
                f"{name} requires SM80+ for this pinned FlashAttention backend"
            )
        root = cluster.root / name
        root.mkdir()
        server = NativeServer(root, device, log_dir / (name + ".log"))
        cluster.servers[name] = server
        server.start()
        cluster.inventory[name] = dict(
            engine="vllm-http",
            model=MODEL,
            revision=REVISION,
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            device=prop.name,
            device_uuid=str(getattr(prop, "uuid", device)),
            visible_device=str(device),
            vocab=len(cluster.text.tok),
            batch_size=1,
            dtype="float16",
            block_size=BLOCK,
            workload="chat-template-v1",
            attention_backend="FLASH_ATTN",
        )
    for name, server in cluster.servers.items():
        cluster.inventory[name]["vllm"] = server.ready(end)
