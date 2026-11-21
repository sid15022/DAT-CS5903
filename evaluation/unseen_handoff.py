"""Run from any directory using the project's CPU dependencies."""
import json
from pathlib import Path
import random
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "DAT_CS5903_Final/DAT-CS5903"))
from failslow.collect import move_kv
from failslow.engines import TinyAttention, cache_key

with tempfile.TemporaryDirectory() as temp:
    prefill, decode = [Path(temp) / name for name in ("prefill", "decode")]
    prefill.mkdir()
    decode.mkdir()
    source, target = TinyAttention(prefill), TinyAttention(decode)
    rng = random.Random(59317)
    prompt = [rng.randrange(8, 250) for _ in range(83)]
    reference = list(source.generate("ref", prompt, 9, "reference", prompt))
    first = list(source.generate("p", prompt, 1, "prefill", prompt))
    key = cache_key(prompt)
    _, _, size, _ = move_kv(prefill/key, decode/key, 0, (0, 0), 1)
    output = first + list(target.generate("d", prompt + first, 8, "decode", prompt))
    loaded = json.loads((decode/key/"loaded.json").read_text())["loaded_tokens"]
    if output != reference or size <= 0 or loaded != len(prompt):
        raise ValueError("Handoff failed the unseen valid case")
    print("PASS: unseen 83-token CPU fixture, 9 output IDs", output)
