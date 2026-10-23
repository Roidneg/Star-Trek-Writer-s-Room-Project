# Star Trek Writer's Room

I’m building a Python writing system that separates creative generation from the checks needed to trust its output. The project explores specialist agents, source provenance, structured outputs, and deterministic workflow controls through Star Trek fan fiction.

The Streamlit demo turns a scene brief into provisional character notes and one screenplay scene. It uses the existing drafting workflow, with **at most two provider calls per submission and no automatic retries**. It does not claim that generated material is canon-verified.

## Demo capabilities

- Explore an illustrative sample without an account, API key, or model call. The sample is authored for the interface, not a recorded successful model run.
- Edit the era, cast, premise, objective, style, requirements, and prohibited assumptions.
- Validate a brief without making a provider call.
- Generate provisional character guidance and a scene using Cloudflare Workers AI or Groq.
- Optionally attach an intact source-package JSON exported by the project. Integrity checks preserve passages and attribution; they do not authenticate an uploaded source or establish that it supports every claim.
- Download Markdown with review notes and a JSON draft package with the generation trace.
- Keep drafts in the current Streamlit session. The app does not write visitor drafts to shared checkpoint files.

The three presets supply separate scene briefs. Choosing a later preset does not automatically continue a previous generated scene. For continuation, edit the premise to include the previous scene's relevant events.

## Run locally

Python **3.12** is the tested deployment version. The original project was developed on Python 3.13; this release has not been independently tested on 3.13.

From the repository root, in a Python 3.12 environment:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The sample and brief validation work without secrets. To preserve an older development environment, create a separate `.venv-demo` environment before installing the pinned dependencies.

For live drafting, copy `.env.example` to `.env` and set your actual values locally:

```dotenv
MODEL_PROVIDER=cloudflare
CLOUDFLARE_API_TOKEN=your-token
CLOUDFLARE_ACCOUNT_ID=your-account-id
WRITERS_ROOM_ACCESS_CODE=a-long-random-demo-code
```

The access code protects the host's quota; it is **not** the provider API token. Share it only with intended live-demo users. Do not enter provider credentials into the app's access-code field.

To use Groq for the draft demo, set `MODEL_PROVIDER=groq` and `GROQ_API_KEY`. All six logical model defaults live in `config.py`; the provider adapter maps `openai/gpt-oss-120b` to Cloudflare's `@cf/openai/gpt-oss-120b`.

Credentials stay outside Git. The application reads root-level Streamlit secrets or the local environment. Live submissions also require a manual-review acknowledgment, run one at a time per server process, and have a 30-second cooldown per session. These controls are not a persistent account-wide spending limit and do not guarantee that a provider's free allocation will last.

## Deploy on Streamlit Community Cloud

After pushing these files to GitHub, select the following in **Create app**:

| Setting | Value |
| --- | --- |
| Repository | `Roidneg/Star-Trek-Writer-s-Room-Project` |
| Branch | The branch containing `app.py` (normally `main`) |
| Main file | `app.py` |
| Python | `3.12` under Advanced settings |

You can deploy the sample immediately with no secrets. To enable live drafting, put these **root-level TOML values** in Advanced settings → Secrets, using your real values only in that settings form:

```toml
MODEL_PROVIDER = "cloudflare"
CLOUDFLARE_API_TOKEN = "your-token"
CLOUDFLARE_ACCOUNT_ID = "your-account-id"
WRITERS_ROOM_ACCESS_CODE = "a-long-random-demo-code"
```

Do not put these values in `config.py` or commit `.streamlit/secrets.toml`. Restart the app after changing provider configuration. Select **Write a scene**, enter the access code, check the acknowledgment, and generate a scene. If the provider rejects a response or runs out of quota, the app reports the failure without retrying or presenting an incomplete scene as completed.

Deployment references: [Streamlit deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [secrets management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management), and [Cloudflare's compatible API](https://developers.cloudflare.com/workers-ai/configuration/open-ai-compatibility/).

## Validation

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
python run_draft_scene.py
```

The default pytest configuration collects only `tests/test_*.py`. Network access is blocked in that suite. Tests exercise input constraints, exact source packages, failure paths, session isolation, Streamlit interactions, and both real provider SDKs against simulated HTTP responses. The CLI command above validates the default brief without model calls; adding `--live` deliberately enables calls.

Older standalone tests remain in `agents/` and `workflows/`. Several need fixture updates, and others intentionally call live models. A green default suite does not certify all legacy research and production workflows. See [the review report](docs/REVIEW_2026-10-23.md) for the baseline inventory and unresolved findings.

## Architecture and current scope

| Component | Responsibility | Used by Streamlit demo? |
| --- | --- | --- |
| `app.py` | Brief editor, sample, session state, review display, downloads | Yes |
| `workflows/draft_writers_room.py` | Two bounded generation stages and structural validation | Yes |
| `tools/model_provider.py` | Cloudflare/Groq client construction | Live drafting only |
| `tools/lore_evidence.py` | Exact source passages and provenance checks | For attached sources |
| Story Architect and Lore agents | Plan development and Memory Alpha research | No |
| Character and Scriptwriter production agents | Evidence-grounded analysis, generation, and semantic audits | No; the demo uses the separate provisional drafting path |
| Continuity and Evaluation agents | State tracking and production review | No |
| Recovery and refresh workflows | Checkpoint restoration and evidence invalidation | No |

The larger production pipeline remains experimental. Its research, per-claim audits, and repairs can require many more calls than the demo. The two-call limit applies to **one drafting submission**, not the entire project.

## Known limits

Model accuracy, live provider access, and free-tier capacity require a real deployment smoke test with the host's credentials. The implementation was validated offline; there is no claim of a successful live generation from this review.

Canon, chronology, character voice, scene continuity, and compliance with prohibited assumptions still require human review. A valid schema is not proof of story quality. Download drafts before closing the session; refreshing the browser, a server restart, or session expiry may discard them.

This is an educational fan project, unaffiliated with Star Trek or its rights holders.
