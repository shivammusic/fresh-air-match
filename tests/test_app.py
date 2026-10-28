import json
import re
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import MiniLMEmbedder, load_activities, rank_activities, validate_request  # noqa: E402


class KeywordEmbedder:
    def embed(self, texts):
        tokens = [set(re.findall(r"[a-z]+", text.lower())) for text in texts]
        vocabulary = sorted(set().union(*tokens))
        return np.asarray([[float(word in row) for word in vocabulary] for row in tokens], dtype=np.float32)


class FreshAirMatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.activities = load_activities()
        cls.embedder = KeywordEmbedder()

    def request(self, **changes):
        payload = {
            "minutes": 30,
            "energy": "low",
            "mobility": "step_free",
            "setting": "park",
            "social": "solo",
            "weather": "clear",
            "note": "quiet trees sketching nature",
        }
        payload.update(changes)
        return payload

    def test_catalogue_ids_are_unique(self):
        self.assertEqual(len(self.activities), len({item["id"] for item in self.activities}))
        self.assertGreaterEqual(len(self.activities), 15)

    def test_request_validation_rejects_boolean_minutes(self):
        with self.assertRaisesRegex(ValueError, "minutes"):
            validate_request(self.request(minutes=True))

    def test_request_validation_limits_private_note(self):
        with self.assertRaisesRegex(ValueError, "500"):
            validate_request(self.request(note="x" * 501))

    def test_hard_filters_are_respected(self):
        result = rank_activities(self.request(minutes=20, mobility="seated", setting="balcony"), self.embedder, self.activities)
        self.assertTrue(result["matches"])
        for match in result["matches"]:
            self.assertIn("seated", match["accessibility"])
            self.assertIn("balcony", match["settings"])
            self.assertLessEqual(match["minutes"], 20)

    def test_partner_filter_never_returns_solo_only(self):
        result = rank_activities(self.request(social="partner", setting="any"), self.embedder, self.activities)
        for match in result["matches"]:
            self.assertIn("partner", match["social"])

    def test_rain_returns_only_rain_safe_catalogue_entries(self):
        result = rank_activities(self.request(weather="light_rain", setting="any"), self.embedder, self.activities)
        for match in result["matches"]:
            self.assertIn("light_rain", match["weather"])

    def test_semantic_ranking_prefers_tree_activity(self):
        result = rank_activities(self.request(note="trees bark nature drawing", setting="park"), self.embedder, self.activities)
        top_ids = {item["id"] for item in result["matches"]}
        self.assertTrue({"tree-bark-notes", "season-change-log"} & top_ids)

    def test_response_makes_privacy_and_safety_explicit(self):
        result = rank_activities(self.request(), self.embedder, self.activities)
        self.assertIn("locally", result["privacy"])
        self.assertIn("not route", result["safety_note"])

    def test_json_catalogue_round_trip(self):
        raw = json.loads((ROOT / "activities.json").read_text(encoding="utf-8"))
        self.assertEqual(raw, self.activities)

    def test_real_model_smoke_when_artifacts_exist(self):
        if not (ROOT / "models" / "model.onnx").exists():
            self.skipTest("Pinned model artifacts are not installed")
        result = rank_activities(self.request(), MiniLMEmbedder(), self.activities)
        self.assertEqual(result["model"], "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(len(result["matches"]), 3)


if __name__ == "__main__":
    unittest.main()

