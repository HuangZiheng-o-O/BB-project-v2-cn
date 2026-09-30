# BB Project — Clinical Evidence Review Agent

This Python CLI creates a source-audited abstraction of text clinical records and answers related questions through a bounded, tool-using agent. It is a prototype for one review corpus at a time. Source records remain read-only; every run gets a new output directory.

**Documentation:** [Runbook: every command and the Gradio page](doc/RUNBOOK.md) · [Submission guide: abstraction, logs, checks and measurements](doc/SUBMISSION.md) · [Implementation walkthrough and related questions](doc/IMPLEMENTATION_WALKTHROUGH.md).

## Architecture and reviewed answers

**[Project homepage](https://huangziheng-o-o.github.io/BB-project/)** · **[Interactive architecture](https://huangziheng-o-o.github.io/BB-project/architecture.html)** · [View its source specification](architecture/bb-project.architecture.json) · [Browse all 27 reviewed question, answer, and evidence reports](validated-answers/README.md)

The diagram was generated and checked with [Archify](https://github.com/tt-a1i/archify). Open the interactive page to explore its components and verified source links.

[![Architecture preview](architecture/bb-project-preview.png)](https://huangziheng-o-o.github.io/BB-project/architecture.html)

| Original question | Reviewed result | Full answer and evidence |
|---|---|---|
| Therapy encounters | 12 sessions across 11 days: 5 individual, 5 group, 2 family | [DEV-01](validated-answers/DEV-01.md) |
| Delivered therapy time | 585–595 minutes overall; weekly totals 140, 120, 180, and 145–155 | [DEV-02](validated-answers/DEV-02.md) |
| Weekly plan goal | Not met, not met, met, and undetermined across the four weeks | [DEV-03](validated-answers/DEV-03.md) |
| January 19 and 21 care | January 19: two contacts, 90 minutes; January 21: one contact, 45 minutes | [DEV-04](validated-answers/DEV-04.md) |
| Symptom course | Three distinct PHQ-9 scores of 18, 14, and 10; partial improvement with ongoing difficulty | [DEV-05](validated-answers/DEV-05.md) |

### Independent questions

| Question | Reviewed result | Full answer and evidence |
|---|---|---|
| 01 · Group authorization | 8 authorized; 5 attended | [Read](validated-answers/INDEPENDENT-01.md) |
| 02 · Family encounters | 2 patient-present encounters; 75 minutes | [Read](validated-answers/INDEPENDENT-02.md) |
| 03 · January 26 records | 1 encounter; 40–50 minutes unresolved | [Read](validated-answers/INDEPENDENT-03.md) |
| 04 · Two skills groups | 45 minutes each; 90 combined | [Read](validated-answers/INDEPENDENT-04.md) |
| 05 · Corrected roster | January 19 group ends at 11:15; 60 therapy minutes | [Read](validated-answers/INDEPENDENT-05.md) |
| 06 · Video reconnection | 1 encounter; 45 minutes | [Read](validated-answers/INDEPENDENT-06.md) |
| 07 · Family support | Planned check-ins improved reported interactions; durable outcome unestablished | [Read](validated-answers/INDEPENDENT-07.md) |
| 08 · Coordination and medication | No employment, therapy-frequency, or medication change documented | [Read](validated-answers/INDEPENDENT-08.md) |
| 09 · Skills-group content | Five themes documented; in-group practice does not prove home completion | [Read](validated-answers/INDEPENDENT-09.md) |
| 10 · January 8 no-show | Later attended treatment rules out stopping participation then | [Read](validated-answers/INDEPENDENT-10.md) |
| 11 · January 15 group | Clinic cancellation; no group held | [Read](validated-answers/INDEPENDENT-11.md) |
| 12 · January 22 group | Late arrival; 45 patient therapy minutes | [Read](validated-answers/INDEPENDENT-12.md) |
| 13 · Work calendar | January 22 example; January 29 reported action, with follow-up delayed | [Read](validated-answers/INDEPENDENT-13.md) |
| 14 · January 19 | 2 encounters, 1 therapy day, 90 minutes | [Read](validated-answers/INDEPENDENT-14.md) |
| 15 · January 19–25 week | 4 encounters, 3 days, 180 minutes; goal met | [Read](validated-answers/INDEPENDENT-15.md) |
| 16 · January 26 week | 3 days, 145–155 minutes; minute goal undetermined | [Read](validated-answers/INDEPENDENT-16.md) |
| 17 · January 26 receipts | Copies preserve earlier events; neither creates new care or assessment | [Read](validated-answers/INDEPENDENT-17.md) |
| 18 · Medication visits | 45 medication-management minutes; 0 psychotherapy-goal minutes | [Read](validated-answers/INDEPENDENT-18.md) |
| 19 · Symptom scores | 18, 14, 10 observed; psychotherapy causation unestablished | [Read](validated-answers/INDEPENDENT-19.md) |
| 20 · Future services | Recommendations do not establish later delivered visits | [Read](validated-answers/INDEPENDENT-20.md) |
| 21 · Employment transition | Preparatory actions documented; no workplace determination | [Read](validated-answers/INDEPENDENT-21.md) |
| 22 · January 30 ledger | 1 family therapy day, 30 patient psychotherapy minutes | [Read](validated-answers/INDEPENDENT-22.md) |

The independent answer key was held outside the repository and was not part of the searchable corpus.

The submitted five-question run used `gpt-6-sol` through the `openai` adapter with a validated reusable abstraction, the January 5–30 review window, and default limits of 12 tool calls and six model turns per question. Its online stage took 101.77 seconds and 17 model calls, recording 136,419 input and 6,288 output tokens; [run metadata and traces](artifacts/reviewed-development/README.md) preserve the details. The matching offline-source run took 214.49 seconds including one question, so that number is not an offline-only duration. Provider billing rates were unavailable, so no reliable dollar cost is reported. Current input support is UTF-8 `.txt`; a PDF/DOCX parser preserving source anchors is the next input-format investigation.

## Setup

Python 3.11+ and `uv` are recommended:

```bash
uv sync
```

Set a key for a **general-purpose, programmatic model API** in the environment. For Z.AI's regular OpenAI-compatible API, use `ZAI_API_KEY`; for OpenAI, use `OPENAI_API_KEY`. The Anthropic adapter accepts `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN`. Keep credentials outside version control and run outputs. A local `.env` file is ignored by Git; load it with `uv run --env-file .env`. Both keys may be set at once: an explicit Z.AI base URL selects `ZAI_API_KEY`, while the default OpenAI endpoint selects `OPENAI_API_KEY`. The [Z.AI Coding Plan endpoint](https://docs.z.ai/devpack/quick-start) is documented for supported coding tools, while the [standard API endpoint](https://docs.z.ai/guides/overview/quick-start) is for programmatic model calls; use the latter for this application with an eligible API key.

For `glm-4.7`, the adapter disables thinking by default to keep extraction and tool responses within the output budget. Set `BB_GLM_THINKING=enabled` to test the reasoning variant. Z.AI documents this [per-turn thinking control](https://docs.z.ai/guides/capabilities/thinking-mode).

## Run

```bash
uv run --env-file .env bb-review \
  --documents data \
  --questions questions.json \
  --provider openai \
  --model gpt-6-sol \
  --start 2026-01-05 --end 2026-01-30
```

The questions file is a JSON array of strings or `{ "id": "...", "question": "..." }` objects. `--start` and `--end` define an inclusive review period; omit them to include all extracted dates. Results appear in a new `runs/<timestamp>-<random>/` directory:

- `abstraction.json`: source hashes, extracted claims, event reconciliation, unresolved items and audit findings.
- `calculation.json`: event ledger, bounded minutes and sessions, weekly totals and goal decisions.
- `answers.json`: answers, citations and citation audit results.
- `reports/question-001.md` and later numbered files: question, answer, exact cited source lines, and online model call count.
- `trace.jsonl`: extraction, reconciliation, model and tool calls with token usage.
- `run.json`: model, runtime, source manifest, stage-specific call counts and usage summary.

Markdown reports contain **Question**, **Answer**, and **Evidence** sections. The model generates the answer and inline citations. The Evidence section copies the cited original document lines locally, without another model call or additional model output tokens. The displayed call count includes only the online calls made to answer that question. Offline processing calls remain separately recorded in `run.json`.

The supplied inputs are `data/` and `questions.json`.

For a new model, omit `--snapshot` to process the original documents. After a successful offline run, later questions with the same model and unchanged documents may reuse its `abstraction.json` through `--snapshot`; source hashes, model identity, and validation findings are checked before reuse. `--reuse-cache` is an explicit content-addressed stage option.

When `--reuse-cache` is explicitly selected, validated extraction and reconciliation batches are cached by model, prompt and input content under the chosen output root's `_stage_cache/`. Progress messages identify the active batch or question.

## Local question page

First let the new model process the original documents, with a separate output root. `--prepare-only` skips the five development answers so you can ask only the questions you want in the page. The command prints a unique run directory when finished:

```bash
PREP_RUN="$(uv run --env-file .env bb-review \
  --documents data --prepare-only \
  --provider openai --model gpt-6-sol \
  --output runs/new-model \
  --start 2026-01-05 --end 2026-01-30)"
printf 'Prepared run: %s\n' "$PREP_RUN"
```

Install the optional Gradio interface and point it to **that new run directory**:

```bash
uv sync --extra web
uv run --env-file .env --extra web bb-review-web \
  --run "$PREP_RUN" \
  --provider openai --model gpt-6-sol
```

Open `http://127.0.0.1:7860`, enter a new question, and click **Ask**. The page shows the answer with source references. **Download Markdown** provides the question, answer, cited original lines, and online model call count. The page checks the model, source hashes, calculation, and validation findings before using a saved offline result. It reuses document processing across questions. Each answer and its model trace are saved under ignored `runs/web/`. To generate the original five answers, run `bb-review` separately with `--questions questions.json` and `--snapshot "$PREP_RUN/abstraction.json"` after the offline run. For ready-to-copy commands, see the [runbook](doc/RUNBOOK.md).

The page listens on the local computer at `127.0.0.1`. For another document set, pass its original source directory with `--documents` to both commands and use the run directory created from those documents.

## Method and checks

The source adapter assigns stable document IDs and line numbers and builds a SQLite FTS5 index. A model extracts source-specific event, plan, measurement and observation claims in batches. Source-line labels are normalized to integer anchors when unambiguous. Extraction, time-scope audit, and reconciliation validate complete model outputs and provide concrete feedback for bounded model repair. A source-coverage check revisits explicitly identified encounters. A selective time-scope audit distinguishes group/session time from patient-specific contact time. A reconciliation pass combines claims about the same encounter, retaining explicit corrections and conflicting intervals. The deterministic calculator unions actual patient-contact intervals, subtracts breaks, aggregates by day and Monday–Sunday week, and compares documented goals. The answer agent can search, open sources, inspect all claims for an event, page through complete inventories and request calculated views.

The code uses a narrow `ModelPort` (Adapter pattern), `Corpus` source/search port (Repository pattern), and a bounded tool runtime (Agent/Command pattern). The model and storage adapters can be replaced independently. Native `anthropic` and `openai` SDKs provide the wire protocols; Pydantic validates the data contracts. This small tool loop keeps the take-home project runnable without a LangGraph service dependency. LangGraph's `create_agent` is an optional replacement for the same tool protocol if a larger deployment needs middleware or durable execution.

The checks cover source addressing, interval union/subtraction, uncertainty bounds, goal comparison, model-stage repair, and report generation. Model-based evaluation uses the supplied questions and independent questions with a reusable offline snapshot.

## Debugging a run

Start with `run.json` to see source hashes, whether the offline snapshot was reused, runtime, token usage, and separate offline and online model-call counts. `abstraction.json` contains extracted claims, reconciled events, and audit findings; `calculation.json` contains the event ledger and weekly totals. For an answer, find its `question_id` entries in `trace.jsonl`: model turns include returned text, stop reason, requested tools, and usage; tool entries record arguments and results. Compare the final `answers.json` entry with its `reports/question-###.md` source excerpts. The Gradio page writes the same answer trace beside each downloaded Markdown report under ignored `runs/web/`.

These local trace and run files can contain source text, questions, and model responses. The `runs/` directory is excluded from Git.

I directed AI-assisted research and led the architecture, discussing the detailed design with AI. AI implemented the project, and I worked with AI to review and test it. Technical ideas and libraries are credited in [the implementation plan](doc/IMPLEMENTATION_PLAN.md).
