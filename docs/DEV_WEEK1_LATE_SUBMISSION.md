# Fresh Air Match: local AI that gets you outside without tracking you

> **Late-review disclosure:** I built this project after the public Week 1 deadline. Hacktoberfest Support authorized an individual late-submission review by email. I am sharing it for that review, not claiming eligibility for the concluded prize judging.

Fresh Air Match turns a short description of your mood into three small outdoor activities you can actually do. The screen is intentionally the shortest part of the experience: choose your constraints, receive a bounded three-step plan, then put the device away.

![Fresh Air Match showing three ranked outdoor activities](https://raw.githubusercontent.com/shivammusic/fresh-air-match/main/docs/fresh-air-match-demo.png)

## What I built

The app asks for:

- the time available;
- weather and preferred setting;
- solo or shared company;
- energy level;
- access needs such as seated or step-free activities; and
- a short free-text note about what sounds restorative.

It filters a curated catalogue of 16 activities using the hard constraints first. Only the eligible activities reach the semantic-ranking stage. A local `sentence-transformers/all-MiniLM-L6-v2` ONNX model then compares the note with those activities and returns the three closest matches.

The catalogue includes neighbourhood, park, doorstep, balcony, rain-friendly, seated, step-free, solo, and partner options. Every result has a time bound, a three-step plan, a short bring list, and a safety reminder.

## Why open-source AI is the core, not decoration

MiniLM is doing the part that ordinary filters cannot: interpreting phrases such as “quiet trees, sketching, and mindful observation” and ranking conceptually related activities even when the wording differs.

The model runs locally through ONNX Runtime. Preference text is not sent to a hosted model, and the app asks for no precise location. The model revision is pinned in the setup script, so the inference component can be inspected or replaced instead of being hidden behind a closed endpoint.

Open innovation matters here for three practical reasons:

1. **Privacy:** a mood or access preference can stay on the user's computer.
2. **Control:** the model can rank only a transparent, curated catalogue.
3. **Reproducibility:** the model revision, conversion path, tests, and limits are all visible.

## Safety and accessibility are deterministic

Semantic similarity never overrides a hard constraint. The server first removes activities that do not fit the requested time, company, weather, setting, energy, or access requirements. AI ranks only what remains.

That design also keeps the model from inventing trails, species, weather conditions, or accessibility claims. Fresh Air Match is not a route planner, medical tool, or personal-safety service. Users still need to assess current conditions and choose familiar public places.

## Architecture

```text
browser form
    |
    v
validated constraints ---> deterministic safety/access filters
                                |
                                v
free-text mood ---> local MiniLM embeddings ---> cosine ranking
                                |
                                v
                    three bounded activity cards
```

The application is a dependency-light Python HTTP server with a plain HTML, CSS, and JavaScript interface. Model weights remain outside Git and are downloaded by a revision-pinned setup script.

## Testing it for real

I ran ten unit and integration tests covering:

- input validation and length limits;
- catalogue integrity;
- time, weather, company, setting, energy, and accessibility filters;
- semantic ranking;
- privacy and safety statements; and
- real ONNX inference when the model artifacts are present.

All ten tests passed. The real-model smoke test passed, JavaScript syntax validation passed, and live HTTP checks returned `200` for the app, health endpoint, and matching endpoint. The genuine screenshot above was captured from the locally running application after it returned three ranked results.

## Try it

Source code and reproducible setup instructions: [github.com/shivammusic/fresh-air-match](https://github.com/shivammusic/fresh-air-match)

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe setup_model.py
.\.venv\Scripts\python.exe app.py
```

Then open `http://127.0.0.1:8766`. Add `?demo=1` to populate and run the example shown in the screenshot.

## What I learned

Two related DEV projects helped sharpen the boundary between useful local AI and unreliable generation:

- Mohamed Adib's [on-device Gemma 2 outdoor planner](https://dev.to/mohamedadib/touch-grass-a-30-second-outdoor-planner-that-runs-gemma-2-inside-your-browser-gdb) demonstrates the value of constrained choices and deterministic safety rules.
- John Gnanprakash Matta's [offline Grass-CLI](https://dev.to/john_gnanprakashmatta_c/grass-cli-an-offline-local-first-ai-terminal-companion-for-outdoor-trail-adventure-itineraries-31gn) shows why offline inference is compelling for outdoor tools.

Fresh Air Match takes a deliberately narrower path: it ranks curated micro-adventures and avoids generating routes or unverified outdoor facts. That smaller scope made privacy, accessibility filtering, and reproducibility much easier to defend.

## Final reflection

“Touch Grass” works best when the technology knows when to disappear. Open-source local inference gives this project enough intelligence to understand intent while keeping the user's preferences private and the suggested actions bounded. The result is not an endless chat—it is a quick nudge toward a real-world activity.
