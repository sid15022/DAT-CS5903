"""Worker-side evidence that vLLM 0.10.2 completed an external KV load."""

import json
from pathlib import Path

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.shared_storage_connector import (
    SharedStorageConnector,
)


class AuditedStorageConnector(SharedStorageConnector):
    def start_load_kv(self, forward_context, **kwargs):
        meta = self._get_connector_metadata()
        super().start_load_kv(forward_context, **kwargs)
        if meta is None or forward_context.attn_metadata is None:
            return
        for req in meta.requests:
            if req.is_store or not req.slot_mapping.numel():
                continue
            torch.cuda.synchronize()
            folder = Path(
                self._generate_foldername_debug(req.token_ids, req.mm_hashes, False)
            )
            (folder / "loaded.json").write_text(
                json.dumps({"loaded_tokens": req.slot_mapping.numel()})
            )


class ServerTimingConnector(AuditedStorageConnector):
    """Single-sequence vLLM V1 forward timing for a private instrumented server."""

    def build_connector_meta(self, scheduler_output):
        metadata = super().build_connector_meta(scheduler_output)
        # The single-device uniprocess executor passes this object to its worker.
        metadata.dat_scheduled_ids = tuple(scheduler_output.num_scheduled_tokens)
        return metadata

    def start_load_kv(self, forward_context, **kwargs):
        import time

        self._dat_active = None
        meta = self._get_connector_metadata()
        ids = getattr(meta, "dat_scheduled_ids", ())
        if ids and forward_context.attn_metadata is not None:
            if len(ids) != 1:
                raise RuntimeError(
                    "This per-request instrument requires max_num_seqs=1"
                )
            control = Path(self._storage_path) / "active.json"
            spec = json.loads(control.read_text())
            expected = "cmpl-" + spec["request_key"] + "-0"
            if ids[0] != expected and not ids[0].startswith(expected + "-"):
                raise RuntimeError(f"Uncorrelated vLLM forward {ids[0]!r}")
            torch.cuda.synchronize()
            self._dat_active = (spec, time.monotonic())
        super().start_load_kv(forward_context, **kwargs)

    def wait_for_save(self):
        import time
        from .workers import delay

        super().wait_for_save()
        active = getattr(self, "_dat_active", None)
        if active is None:
            return
        spec, start = active
        self._dat_active = None
        torch.cuda.synchronize()
        injected = delay(start, spec["origin"], spec["interval"], spec["factor"])
        end = time.monotonic()
        with (Path(self._storage_path) / "forward.jsonl").open("a") as sink:
            sink.write(json.dumps([spec["request_key"], start, end, injected]) + "\n")
