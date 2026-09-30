# Runbook

This guide runs the command-line reviewer, the local Gradio question page, and the inspection tools. Run commands from the repository root. The supplied corpus is `data/`; `questions.json` contains the five development questions. The program accepts a directory of UTF-8 `.txt` files and does not contain patient-specific answer rules.

## 1. Install and configure a model

Install Python 3.11 or newer and [`uv`](https://docs.astral.sh/uv/). Then run:

```bash
uv sync --locked
```

Create a local `.env` file in the repository root. It is ignored by Git. Add the key for the provider you use; never commit real credentials:

```dotenv
# Use one or more of these, as appropriate.
OPENAI_API_KEY=your_openai_key
ZAI_API_KEY=your_zai_standard_api_key
ANTHROPIC_API_KEY=your_anthropic_key
```

The `openai` adapter selects `ZAI_API_KEY` when its explicit base URL has host `api.z.ai`; otherwise it selects `OPENAI_API_KEY`. The `anthropic` adapter also accepts `ANTHROPIC_AUTH_TOKEN` and `ANTHROPIC_BASE_URL`. An optional `BB_MODEL_TIMEOUT_SECONDS` controls the provider request timeout (default: 180 seconds). For `glm-4.7`, thinking is disabled by default; `BB_GLM_THINKING=enabled` opts in. [Z.AI's standard API guide](https://docs.z.ai/guides/overview/quick-start) documents the OpenAI-compatible base URL used below.

Use one of these model options consistently in both the preparation and question commands:

| Service | Command options | Key |
| --- | --- | --- |
| OpenAI | `--provider openai --model gpt-6-sol` for the reviewed example | `OPENAI_API_KEY` |
| Z.AI standard API | `--provider openai --model glm-4.7 --base-url https://api.z.ai/api/paas/v4/` | `ZAI_API_KEY` |
| Anthropic or compatible endpoint | `--provider anthropic --model YOUR_MODEL` with optional `--base-url` | `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` |

Model availability and billing depend on the account. The repository's reviewed development bundle used `gpt-6-sol`; you may use another model available to you, but a snapshot is reusable only with the same model name and unchanged source hashes.

## 2. Answer the supplied five questions from fresh source processing

With an OpenAI key that can access `gpt-6-sol`, run the same model used for the reviewed examples:

```bash
uv run --env-file .env bb-review \
  --documents data \
  --questions questions.json \
  --provider openai --model gpt-6-sol \
  --start 2026-01-05 --end 2026-01-30
```

For Z.AI's standard API, use:

```bash
uv run --env-file .env bb-review \
  --documents data \
  --questions questions.json \
  --provider openai --model glm-4.7 \
  --base-url https://api.z.ai/api/paas/v4/ \
  --start 2026-01-05 --end 2026-01-30
```

The command prints a unique directory such as `runs/20260930T061914Z-3aa79c`. It processes the corpus offline, calculates the review, and then answers each question online. `--start` and `--end` set an inclusive calculation window; omit both to cover all extracted dates. `--output runs/my-review` changes the parent of new run directories.

### Run only the reusable offline preparation

```bash
PREP_RUN="$(uv run --env-file .env bb-review \
  --documents data --prepare-only \
  --provider openai --model gpt-6-sol \
  --start 2026-01-05 --end 2026-01-30)"
printf 'Prepared run: %s\n' "$PREP_RUN"
```

The captured directory contains `abstraction.json` and `calculation.json`; `answers.json` is an empty array. In the same shell, answer the five questions without another offline model pass:

```bash
uv run --env-file .env bb-review \
  --documents data \
  --questions questions.json \
  --provider openai --model gpt-6-sol \
  --snapshot "$PREP_RUN/abstraction.json" \
  --start 2026-01-05 --end 2026-01-30
```

If you want to skip fresh preparation, replace `"$PREP_RUN/abstraction.json"` with the checked-in `artifacts/reviewed-development/abstraction.json`. This bundled snapshot was produced with `gpt-6-sol` from the same 31 files in `data/`.

The reviewer checks model identity and every source hash before reusing a snapshot. It recalculates the requested date window from the saved abstraction. Change or add a source document, change the model, or need a fresh extraction? Run again without `--snapshot`.

### Ask different questions in a batch

Create a JSON array outside the original `questions.json`, under the ignored `runs/` directory. This complete command creates a unique question file and then uses the checked-in `gpt-6-sol` snapshot:

```bash
mkdir -p runs
QUESTIONS_FILE="$(mktemp runs/questions-XXXXXX.json)"
cat > "$QUESTIONS_FILE" <<'JSON'
[
  {"id": "NEW-01", "question": "What does the record establish about the January 26 individual encounter?"},
  {"id": "NEW-02", "question": "Which symptom assessments are distinct rather than received copies?"}
]
JSON

uv run --env-file .env bb-review \
  --documents data \
  --questions "$QUESTIONS_FILE" \
  --provider openai --model gpt-6-sol \
  --snapshot artifacts/reviewed-development/abstraction.json \
  --start 2026-01-05 --end 2026-01-30
```

Each item can be a string or an object with `id` and `question`. The output reports retain the exact question, answer, inline citations, copied original evidence lines, and the number of **online** model calls for that question.

## 3. Use the Gradio page for one new question at a time

Install the optional interface:

```bash
uv sync --locked --extra web
```

To try the page immediately, use the checked-in preparation bundle. The model name and corpus must match that bundle:

```bash
uv run --env-file .env --extra web bb-review-web \
  --documents data \
  --run artifacts/reviewed-development \
  --provider openai --model gpt-6-sol
```

For a fresh `gpt-6-sol` preparation, use `--run "$PREP_RUN"` after the preparation command in Section 2. For Z.AI, a `glm-4.7` preparation and page launch can be run together:

```bash
ZAI_PREP_RUN="$(uv run --env-file .env bb-review \
  --documents data --prepare-only \
  --provider openai --model glm-4.7 \
  --base-url https://api.z.ai/api/paas/v4/ \
  --start 2026-01-05 --end 2026-01-30)"
uv run --env-file .env --extra web bb-review-web \
  --documents data --run "$ZAI_PREP_RUN" \
  --provider openai --model glm-4.7 \
  --base-url https://api.z.ai/api/paas/v4/
```

Open `http://127.0.0.1:7860`, type a question, and select **Ask**. The answer appears with citations; **Download Markdown** downloads a report containing the question, answer, evidence excerpts, and online call count. The page listens on localhost only. `--port 7861` selects another port; `--output runs/my-web-answers` changes where web answers are saved (default `runs/web/`). Stop it with `Ctrl+C`.

The web session validates the source hashes, model name, saved calculation, and abstraction findings before accepting questions. Each answer receives a unique directory under `runs/web/` with `answer.md`, `trace.jsonl`, and `run.json`. It does not rerun offline extraction for every question.

## 4. Inspect and debug a run

| File | What to inspect |
| --- | --- |
| `abstraction.json` | Source hashes, extracted claims, linked encounters, competing mentions, audit findings. |
| `calculation.json` | Event-level inclusion, minute bounds, session/day counts, weekly results and goal decisions. |
| `answers.json` | Question IDs, answer text, citations, citation audit, online model calls. |
| `reports/question-001.md` | Reviewer-friendly question, answer and exact original lines. |
| `trace.jsonl` | Offline model/validation steps and online model/tool turns; answer entries carry `question_id`. |
| `run.json` | Provider, model, runtime, source manifest, stage call counts, token use and cost status. |

For example:

```bash
uv run python -m json.tool artifacts/reviewed-development/run.json
uv run python -m json.tool artifacts/reviewed-development/calculation.json
rg '"question_id": "DEV-02"|"stage": "citation_audit"' artifacts/reviewed-development/trace.jsonl
```

For a wrong answer, locate its `question_id` in the trace, identify the tool results it saw, then check the underlying source lines and the abstraction's competing claims. `search` is ranked and is not an exhaustive count; `scan` pages complete inventories, `related` displays a reconciled event with all linked mentions, `calculate` returns deterministic totals, and `open_source` shows numbered original lines. The agent is bounded by `--max-tool-calls` (default 12) and `--max-model-turns` (default 6). The offline batch size can be adjusted with `--batch-chars` (default 13,500). Validation failures during extraction and reconciliation go back to the model for bounded repair before a snapshot is accepted.

The optional `--reuse-cache` reuses validated extraction and reconciliation batches in the output parent's `_stage_cache/` when their model, prompt, and input content match. It is separate from `--snapshot`, which reuses a complete abstraction.

## 5. Reproduce checks and inspect the supplied example

```bash
uv run python -m unittest discover -s tests -v
```

The submitted five-question `gpt-6-sol` bundle is in [`artifacts/reviewed-development/`](../artifacts/reviewed-development/). Its offline and online traces are separate, and the five polished reports are [`validated-answers/DEV-01.md`](../validated-answers/DEV-01.md) through `DEV-05.md`. The independent report index is [`validated-answers/README.md`](../validated-answers/README.md). The live [architecture page](https://huangziheng-o-o.github.io/BB-project/architecture.html) links components to source code.

`runs/` and `.env` are Git-ignored because local traces may contain source text and credentials must stay private. The bundled records are fictional; do not assume this repository's logging configuration is appropriate for real patient data.
