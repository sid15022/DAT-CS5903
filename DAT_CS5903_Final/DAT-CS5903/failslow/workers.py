"""A private IPC worker per engine. CUDA is initialized after GPU selection."""

import json
import os
from pathlib import Path
import shutil
import time
import traceback

from .engines import BLOCK, cache_key


def delay(start, origin, interval, factor):
    if factor > 1 and interval[0] <= start - origin < interval[1]:
        time.sleep((factor - 1) * (time.monotonic() - start))
        return True
    return False


def worker(conn, backend, device, root):
    try:
        os.environ.update(
            CUDA_VISIBLE_DEVICES=str(device),
            VLLM_USE_V1="1",
            VLLM_ENABLE_V1_MULTIPROCESSING="0",
            TOKENIZERS_PARALLELISM="false",
            VLLM_ATTENTION_BACKEND="FLASH_ATTN",
            VLLM_KV_CACHE_LAYOUT="NHD",
            VLLM_LOGGING_LEVEL="WARNING",
        )
        from .engines import TinyAttention, VllmEngine

        engine = (VllmEngine if backend == "vllm" else TinyAttention)(root)
        conn.send(("ready", engine.info))
        while True:
            job = conn.recv()
            if job is None:
                return
            if job["mode"] == "prompt":
                ids = engine.sample_prompt() if backend == "vllm" else list(range(129))
                conn.send(("done", ids))
                continue
            ids = job["ids"]
            original = job["original"]
            mode = job["mode"]
            folder = Path(root) / cache_key(original)
            if mode in ("reference", "prefill") and folder.exists():
                raise RuntimeError("Stale prefill cache would contaminate request")
            if mode == "decode" and (
                not folder.is_dir() or (folder / "loaded.json").exists()
            ):
                raise RuntimeError("Missing or previously consumed KV handoff")
            gen = engine.generate(job["rid"], ids, job["count"], mode, original)
            for k in range(job["count"]):
                start = time.monotonic()
                token = next(gen)
                injected = delay(start, job["origin"], job["interval"], job["factor"])
                end = time.monotonic()
                if k == 0 and mode == "decode":
                    proof = json.loads((folder / "loaded.json").read_text())
                    expected = (
                        (len(original) - 1) // BLOCK * BLOCK
                        if backend == "vllm"
                        else len(original)
                    )
                    if expected <= 0 or proof["loaded_tokens"] != expected:
                        raise RuntimeError(
                            f"Expected {expected} external KV tokens, got {proof}"
                        )
                conn.send(("token", (token, k, start, end, injected)))
            try:
                next(gen)
            except StopIteration:
                pass
            else:
                raise RuntimeError("Excess generated tokens")
            proof = (
                json.loads((folder / "loaded.json").read_text())
                if mode == "decode"
                else {}
            )
            if mode in ("reference", "decode") and folder.exists():
                shutil.rmtree(folder)
            conn.send(("done", proof))
    except (KeyboardInterrupt, EOFError, BrokenPipeError):
        return
    except BaseException:
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()
