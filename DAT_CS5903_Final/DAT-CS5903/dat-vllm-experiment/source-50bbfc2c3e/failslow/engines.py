"""Measured worker engines. Imported in isolated single-worker processes."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
BLOCK = 16


def cache_key(ids):
    prefix = (len(ids) - 1) // BLOCK * BLOCK
    return hashlib.md5(
        np.asarray(ids[:prefix], dtype=np.int64).tobytes(), usedforsecurity=False
    ).hexdigest()


class TinyAttention:
    """Deterministic, single-head causal attention with learned-shaped random weights.

    A CPU integration fixture. Its token choices have no language-quality interpretation.
    """

    def __init__(self, root):
        self.root = Path(root)
        rng = np.random.default_rng(717)

        def weight(n, m):
            return (rng.standard_normal((n, m)) / np.sqrt(n)).astype("float32")

        self.embed = weight(257, 32)
        self.q, self.k, self.v = [weight(32, 32) for _ in range(3)]
        self.out = weight(32, 257)
        self.info = dict(
            engine="numpy-causal-attention",
            numpy=np.__version__,
            weights_seed=717,
            vocab=257,
            device="cpu",
            transfer="npz-copy",
            batch_size=1,
        )

    def step(self, x, k, v):
        logits = (x @ self.q) @ k.T / np.sqrt(k.shape[1])
        a = np.exp(logits - logits.max())
        return int(np.argmax(((a / a.sum()) @ v + x) @ self.out))

    def generate(self, rid, ids, count, mode, original):
        x = self.embed[np.array(ids)]
        folder = self.root / cache_key(original)
        if mode == "decode":
            with np.load(folder / "cache.npz", allow_pickle=False) as kv:
                k, v = kv["k"], kv["v"]
            n = len(k)
            k = np.concatenate([k, x[n:] @ self.k])
            v = np.concatenate([v, x[n:] @ self.v])
            (folder / "loaded.json").write_text(json.dumps(dict(loaded_tokens=n)))
        else:
            k = x @ self.k
            v = x @ self.v
            if mode == "prefill":
                folder.mkdir()
                np.savez(folder / "cache.npz", k=k, v=v)
        current = x[-1]
        for _ in range(count):
            token = self.step(current, k, v)
            yield token
            current = self.embed[token]
            k = np.vstack([k, current @ self.k])
            v = np.vstack([v, current @ self.v])


class VllmEngine:
    def __init__(self, root):
        from importlib.metadata import version

        if version("vllm") != "0.10.2":
            raise RuntimeError("This connector targets vllm==0.10.2")
        from vllm import LLM, SamplingParams
        from vllm.config import KVTransferConfig
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable inside worker")
        if torch.cuda.get_device_capability(0)[0] < 8:
            raise RuntimeError(
                "The pinned FlashAttention testbed needs an Ampere-or-newer CUDA GPU (SM80+)"
            )
        self.params = SamplingParams
        self.root = Path(root)
        self.llm = LLM(
            model=MODEL,
            revision=REVISION,
            tokenizer_revision=REVISION,
            dtype="float16",
            enforce_eager=True,
            max_model_len=512,
            block_size=BLOCK,
            max_num_seqs=1,
            max_num_batched_tokens=512,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            gpu_memory_utilization=0.65,
            kv_transfer_config=KVTransferConfig(
                kv_connector="AuditedStorageConnector",
                kv_connector_module_path="failslow.connector",
                kv_role="kv_both",
                kv_connector_extra_config={"shared_storage_path": str(self.root)},
            ),
        )
        prop = torch.cuda.get_device_properties(0)
        self.info = dict(
            engine="vllm",
            vllm=version("vllm"),
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            model=MODEL,
            revision=REVISION,
            device=prop.name,
            device_uuid=str(getattr(prop, "uuid", os.environ["CUDA_VISIBLE_DEVICES"])),
            memory_bytes=prop.total_memory,
            compute_capability=[prop.major, prop.minor],
            vocab=150000,
            batch_size=1,
            attention_backend="FLASH_ATTN",
            block=BLOCK,
            dtype="float16",
            transfer="v0.10.2 audited shared-storage connector; local file copy",
        )

    def sample_prompt(self):
        sentence = "This experiment separates prefill from decode and transfers the attention cache. "
        ids = self.llm.get_tokenizer().encode(
            sentence * 25 + "Name the two stages.", add_special_tokens=False
        )
        return ids[:129]

    def generate(self, rid, ids, count, mode, original):
        params = self.params(
            temperature=0, max_tokens=count, ignore_eos=True, detokenize=False
        )
        engine = self.llm.llm_engine
        engine.add_request(rid, {"prompt_token_ids": ids}, params)
        seen = 0
        while engine.has_unfinished_requests():
            for result in engine.step():
                if result.request_id != rid or not result.outputs:
                    raise RuntimeError("Unexpected engine response")
                ids_out = result.outputs[0].token_ids
                if len(ids_out) == seen:
                    continue
                if len(ids_out) != seen + 1:
                    raise RuntimeError(
                        "Cannot measure individual tokens from multi-token steps"
                    )
                seen += 1
                yield int(ids_out[-1])
        if seen != count:
            raise RuntimeError("Incomplete generation")
