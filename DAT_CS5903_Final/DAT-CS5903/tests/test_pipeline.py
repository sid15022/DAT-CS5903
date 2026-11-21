import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from failslow.collect import move_kv, schedule
from failslow.data import CLOCK, validate
from failslow.detector import Baseline, Monitor, components, windows
from failslow.engines import TinyAttention, cache_key
from failslow.model import CFG
from failslow.study import Trial, design, simulate


class PassiveDiagnosis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parts = {}
        for i, part in enumerate(["fit", "scale", "threshold"]):
            trial = Trial(part, part, 501 + i, "healthy")
            v, _ = simulate(trial, 180)
            parts[part] = [(part, True, 180, v)]
        cls.baseline = Baseline.train(parts, 2, CFG)
        cls.view, _ = simulate(
            Trial("test", "test", 73, "decode", target="d1", strength=3), 180
        )

    def decisions(self, view, until):
        monitor = Monitor(self.baseline)
        return [
            d
            for tick, raw in windows(
                view, until, self.baseline.model, components(2), CFG
            )
            for d in monitor.step("test", tick, raw)[0]
        ]

    def test_labels_cannot_cross_boundary(self):
        v = {
            **self.view,
            "observations": self.view["observations"].assign(true_component="d1"),
        }
        with self.assertRaises(ValueError):
            validate(v, "test")

    def test_duplicates_rejected(self):
        v = dict(self.view)
        v["observations"] = pd.concat([v["observations"], v["observations"].iloc[:1]])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate(v, "test")

    def test_late_arrival_and_future_invariance(self):
        v = dict(self.view)
        v["observations"] = v["observations"].copy()
        v["observations"].loc[v["observations"].stage == "decode", "available_s"] += 38
        early = {
            k: f[(f.available_s <= 90) & (f[CLOCK[k]] <= 90)].copy()
            for k, f in v.items()
        }
        a = pd.DataFrame(self.decisions(v, 90))
        b = pd.DataFrame(self.decisions(early, 90))
        pd.testing.assert_frame_equal(a, b)
        # These decode records arrive after their event-time windows have expired.
        self.assertFalse(a.full_coverage.any())

    def test_timestamp_semantics(self):
        v = dict(self.view)
        v["observations"] = v["observations"].copy()
        v["observations"].loc[0, "available_s"] = 0
        with self.assertRaisesRegex(ValueError, "availability"):
            validate(v, "test")

    def test_missing_evidence_or_unseen_configuration_abstains(self):
        for observed in [
            self.view["observations"].iloc[:0],
            self.view["observations"].assign(config_id="different-hardware"),
        ]:
            out = self.decisions({**self.view, "observations": observed}, 60)
            self.assertTrue(
                all(d["explanation"] == "unknown" and not d["ranked"] for d in out)
            )

    def test_pairs_match_offered_work(self):
        trials = design(3, 180, 2, "vllm")
        self.assertEqual(len({t.rid for t in trials}), len(trials))
        by_id = {t.rid: t for t in trials}
        for t in trials:
            if t.target is not None:
                clean = by_id[t.pair]
                self.assertEqual(
                    schedule(t, 4, 10, 2, 100), schedule(clean, 4, 10, 2, 100)
                )
        self.assertEqual(sum(t.target is not None for t in trials), 36)

    def test_no_peer_in_one_by_one(self):
        _, (scores, _, _) = next(
            windows(self.view, 30, self.baseline.model, components(1), CFG)
        )
        self.assertTrue(all(np.isnan(s["matched_peer"]) for s in scores.values()))

    def test_real_cache_handoff_is_token_equivalent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for c in ["p", "d"]:
                (root / c).mkdir()
            a, b = TinyAttention(root / "p"), TinyAttention(root / "d")
            ids = list(range(65))
            key = cache_key(ids)
            ref = list(a.generate("ref", ids, 6, "reference", ids))
            head = list(a.generate("p", ids, 1, "prefill", ids))
            _, _, size, _ = move_kv(root / "p" / key, root / "d" / key, 0, (0, 0), 1.0)
            result = head + list(b.generate("d", ids + head, 5, "decode", ids))
            self.assertEqual(ref, result)
            self.assertGreater(size, 0)
            self.assertFalse((root / "p" / key).exists())
            self.assertEqual(
                json.loads((root / "d" / key / "loaded.json").read_text())[
                    "loaded_tokens"
                ],
                len(ids),
            )


if __name__ == "__main__":
    unittest.main()
