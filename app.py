"""Fresh Air Match: local semantic matching for safe micro-adventures."""
from __future__ import annotations

import argparse
import json
import math
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Protocol

import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
ALLOWED_ENERGY = {"low", "medium", "high"}
ALLOWED_MOBILITY = {"seated", "step_free", "standard"}
ALLOWED_SETTINGS = {"any", "balcony", "neighbourhood", "park"}
ALLOWED_SOCIAL = {"any", "solo", "partner"}
ALLOWED_WEATHER = {"any", "clear", "cold", "hot", "light_rain"}


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class MiniLMEmbedder:
    """Pinned CPU-only ONNX embedding runtime."""

    def __init__(self, model_dir: Path = ROOT / "models") -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_path = model_dir / "model.onnx"
        tokenizer_path = model_dir / "tokenizer.json"
        if not model_path.exists() or not tokenizer_path.exists():
            raise FileNotFoundError("Model files are missing. Run setup_model.py first.")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.lock = threading.Lock()

    def embed(self, texts: list[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        with self.lock:
            for start in range(0, len(texts), 16):
                batch = self.tokenizer.encode_batch(texts[start : start + 16])
                ids = np.asarray([item.ids for item in batch], dtype=np.int64)
                masks = np.asarray([item.attention_mask for item in batch], dtype=np.int64)
                hidden = self.session.run(
                    None,
                    {
                        "input_ids": ids,
                        "attention_mask": masks,
                        "token_type_ids": np.asarray([item.type_ids for item in batch], dtype=np.int64),
                    },
                )[0]
                expanded = masks[..., None].astype(np.float32)
                pooled = (hidden * expanded).sum(axis=1) / np.maximum(expanded.sum(axis=1), 1e-9)
                pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9)
                rows.extend(pooled)
        return np.asarray(rows)


def load_activities(path: Path = ROOT / "activities.json") -> list[dict]:
    activities = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(activities, list) or len(activities) < 10:
        raise ValueError("Activity catalogue must contain at least ten entries.")
    ids = [item.get("id") for item in activities]
    if len(ids) != len(set(ids)):
        raise ValueError("Activity identifiers must be unique.")
    return activities


def _choice(payload: dict, key: str, allowed: set[str]) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{key} must be one of: {', '.join(sorted(allowed))}")
    return value


def validate_request(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    minutes = payload.get("minutes")
    if isinstance(minutes, bool) or not isinstance(minutes, int) or not 10 <= minutes <= 120:
        raise ValueError("minutes must be an integer from 10 to 120.")
    note = payload.get("note", "")
    if not isinstance(note, str) or len(note.strip()) > 500:
        raise ValueError("note must be text up to 500 characters.")
    return {
        "minutes": minutes,
        "energy": _choice(payload, "energy", ALLOWED_ENERGY),
        "mobility": _choice(payload, "mobility", ALLOWED_MOBILITY),
        "setting": _choice(payload, "setting", ALLOWED_SETTINGS),
        "social": _choice(payload, "social", ALLOWED_SOCIAL),
        "weather": _choice(payload, "weather", ALLOWED_WEATHER),
        "note": note.strip(),
    }


def _mobility_ok(activity: dict, mobility: str) -> bool:
    available = set(activity["accessibility"])
    if mobility == "seated":
        return "seated" in available
    if mobility == "step_free":
        return "step_free" in available or "seated" in available
    return "standard" in available or "step_free" in available or "seated" in available


def _candidate_text(activity: dict) -> str:
    return " ".join(
        [activity["title"], activity["description"], *activity["tags"], *activity["settings"]]
    )


def _query_text(request: dict) -> str:
    return (
        f"I want a {request['energy']} energy outdoor activity for {request['minutes']} minutes. "
        f"Setting: {request['setting']}. Company: {request['social']}. "
        f"Weather: {request['weather']}. Accessibility: {request['mobility']}. "
        f"Interests and mood: {request['note'] or 'open to a calm local micro-adventure'}."
    )


def rank_activities(payload: dict, embedder: Embedder, activities: list[dict] | None = None) -> dict:
    request = validate_request(payload)
    catalogue = activities if activities is not None else load_activities()
    eligible = []
    for activity in catalogue:
        if activity["min_minutes"] > request["minutes"]:
            continue
        if not _mobility_ok(activity, request["mobility"]):
            continue
        if request["setting"] != "any" and request["setting"] not in activity["settings"]:
            continue
        if request["social"] != "any" and request["social"] not in activity["social"]:
            continue
        if request["weather"] != "any" and request["weather"] not in activity["weather"]:
            continue
        eligible.append(activity)
    if not eligible:
        raise ValueError("No safe catalogue activity matches every selected constraint. Try 'any' for setting or company.")

    texts = [_query_text(request), *(_candidate_text(item) for item in eligible)]
    vectors = np.asarray(embedder.embed(texts), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise ValueError("Embedding output does not match the requested texts.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-9)
    semantic = vectors[1:] @ vectors[0]

    scored = []
    for index, activity in enumerate(eligible):
        score = float(semantic[index])
        if activity["energy"] == request["energy"]:
            score += 0.09
        if request["setting"] != "any" and request["setting"] in activity["settings"]:
            score += 0.05
        if request["social"] != "any" and request["social"] in activity["social"]:
            score += 0.04
        duration = min(request["minutes"], activity["max_minutes"])
        scored.append((score, activity, duration))
    scored.sort(key=lambda row: (-row[0], row[1]["title"]))

    matches = []
    for score, activity, duration in scored[:3]:
        match = {key: value for key, value in activity.items() if key not in {"min_minutes", "max_minutes"}}
        match.update(
            {
                "minutes": duration,
                "match_percent": max(1, min(99, round((score + 1) * 45))),
                "why": f"Local semantic similarity plus hard filters for {request['mobility']} access, time, weather, and setting.",
            }
        )
        matches.append(match)
    return {
        "schema": "fresh-air-match/1",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "privacy": "All matching ran locally. No location, preference, or activity data left this process.",
        "safety_note": "Catalogue suggestions are general ideas, not route, weather, accessibility, or medical verification.",
        "eligible_count": len(eligible),
        "matches": matches,
    }


class App:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.activities = load_activities()

    def run(self, payload: dict) -> dict:
        return rank_activities(payload, self.embedder, self.activities)


class Handler(BaseHTTPRequestHandler):
    app: App

    def _headers(self, status: int, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'")
        self.end_headers()

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8")
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        route = self.path.split("?", 1)[0]
        if route == "/api/health":
            self._json(200, {"ok": True, "model": MODEL_ID, "activities": len(self.app.activities)})
            return
        relative = "index.html" if route == "/" else route.lstrip("/")
        target = (ROOT / "static" / relative).resolve()
        static_root = (ROOT / "static").resolve()
        if static_root not in target.parents and target != static_root:
            self._json(404, {"error": "Not found"})
            return
        if not target.is_file():
            self._json(404, {"error": "Not found"})
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._headers(200, f"{content_type}; charset=utf-8" if content_type.startswith("text/") else content_type)
        self.wfile.write(target.read_bytes())

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/match":
            self._json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 16_384:
                raise ValueError("Request body must be between 1 and 16384 bytes.")
            payload = json.loads(self.rfile.read(length))
            self._json(200, self.app.run(payload))
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
        except Exception:
            self._json(500, {"error": "Local inference failed. Check the model files and try again."})

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[fresh-air-match] {self.address_string()} {fmt % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--smoke", action="store_true", help="Run one real-model match and exit.")
    args = parser.parse_args()
    app = App(MiniLMEmbedder())
    if args.smoke:
        result = app.run(
            {
                "minutes": 30,
                "energy": "low",
                "mobility": "step_free",
                "setting": "park",
                "social": "solo",
                "weather": "clear",
                "note": "quiet nature, trees, sketching, and mindful observation",
            }
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    Handler.app = app
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Fresh Air Match is running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

