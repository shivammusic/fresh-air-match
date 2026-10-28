"""Download the pinned open model artifacts after explicit user invocation."""
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

for remote, local in (("onnx/model.onnx", "model.onnx"), ("tokenizer.json", "tokenizer.json")):
    source = hf_hub_download(repo_id=MODEL, filename=remote, revision=REVISION)
    target = ROOT / "models" / local
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(Path(source).read_bytes())
    print(f"Ready: {target.name} ({target.stat().st_size:,} bytes)")

