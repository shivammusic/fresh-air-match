# Fresh Air Match

Fresh Air Match is a private, local-first micro-adventure matcher. It uses the
open `sentence-transformers/all-MiniLM-L6-v2` model through ONNX Runtime to rank
a curated catalogue against a person's mood and interests, after deterministic
filters enforce their time, access, setting, company, and weather choices.

This is a **retrospective build inspired by the concluded Hacktoberfest 2026
Week 1 “Touch Grass” prompt**. It was created after that challenge closed and is
not presented as an eligible contest entry or sticker claim.

## What makes it useful

- The model runs on the user's CPU; preference text never leaves the process.
- It asks for no precise location and makes no route-accessibility claims.
- Hard constraints run before semantic ranking, so AI similarity cannot override
  seated/step-free, time, social, setting, or weather filters.
- Every activity includes a bounded three-step plan and an explicit safety note.
- Sixteen curated activities cover seated, step-free, solo, partner, rain, park,
  neighbourhood, and balcony/doorstep contexts.

## Architecture

```text
browser form
    │
    ▼
validated constraints ──► deterministic safety/access filters
                                │
                                ▼
free-text mood ──► local MiniLM embeddings ──► cosine ranking
                                │
                                ▼
                    three bounded activity cards
```

The model chooses among curated activities; it does not invent routes, species,
weather, or accessibility information.

## Run

```powershell
./Start-FreshAirMatch.ps1
```

Then open <http://127.0.0.1:8766>.

Manual setup:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe setup_model.py
.\.venv\Scripts\python.exe app.py
```

The model downloader pins revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Model weights are intentionally
excluded from Git.

## Verify

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe app.py --smoke
```

The test suite validates input limits, catalogue integrity, all hard filters,
semantic ranking, privacy/safety output, and real ONNX inference when the model
artifacts are installed.

## Honest limitations

- This is an activity matcher, not a routing, accessibility, weather, medical,
  ecological, or personal-safety service.
- The curated catalogue is intentionally small and general.
- MiniLM ranks semantic similarity; it does not generate natural-language plans.
- Users must independently assess conditions and choose familiar public places.

## License

MIT. The model has its own upstream licensing and attribution; see
`THIRD_PARTY_NOTICES.md`.

