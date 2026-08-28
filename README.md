# Haystack CUAD Contract Review with LiteLLM + Witdem

`haystack-cuad-contract-review` is a reference application for building an
observable, multi-model contract workflow with Haystack orchestration,
LiteLLM model routing, optional CUAD ingestion, and Witdem analytics.

## What it does

This reference application reviews fictional-company vendor SaaS agreements through a realistic, observable contract-operations workflow. It selects native PDF extraction or Mistral OCR, uses DeepSeek for text transformations, applies a fictional playbook, asks OpenAI GPT-5.4 to judge extraction quality, risk, fallback acceptability, and the final state, then returns an evidence-backed machine-readable audit and post-signature obligations.

## Architecture

~~~text
Contract (PDF / scan / text)
  -> Haystack Pipeline (31 visible components)
       -> ConditionalRouter: native text or vision
       -> extraction + normalization + clauses + typed terms
       -> GPT judge -> focused retry loop (maximum 2 retries)
       -> fictional playbook -> compliant or deviation branch
       -> risk -> domain routing -> fallback -> judge/retry loop
       -> final GPT gate -> approved or human-review branch
       -> obligations when approved -> Pydantic result validation
  -> ModelRegistry
       -> official litellm-haystack LiteLLMChatGenerator
       -> LiteLLM OCR endpoint for document input
  -> declarative role -> provider/model
~~~

Haystack owns orchestration and branching. LiteLLM owns model/provider access. Components request a logical role rather than a provider. Model routing is declared in [model_routing.yaml](contract_review_agent/app/model_routing.yaml), so model experiments do not require graph changes.

The implementation targets haystack-ai 3.1.x and the maintained litellm-haystack 1.2.x integration. It uses Haystack 3 chat generators and avoids the legacy generators removed in Haystack 3.

## Haystack showcase

This repository is designed to make the Haystack execution model visible, not
to hide the workflow behind one agent call. The graph contains typed business
components, `ConditionalRouter` branches, joiners, bounded retry loops, and one
shared role registry built on the official `litellm-haystack` integration.

The strongest live demo uses a raster-only PDF so one execution visibly crosses
all three model roles:

~~~text
Haystack input and quality routing
  -> Mistral document OCR through LiteLLM
  -> DeepSeek extraction and normalization through LiteLLMChatGenerator
  -> GPT-5.4 quality, risk, and final gates through LiteLLMChatGenerator
  -> Witdem execution graph, retries, tokens, cost, and business outcome
~~~

After the Witdem services are running and `.env` contains the three provider
keys, create a safe fictional scan and run the verifier:

~~~bash
uv run python examples/create_showcase_scan.py
uv run contract-review-showcase output/showcase/scanned-vendor-saas.pdf
~~~

The command exits unsuccessfully unless Witdem exposes the new Haystack run and
attributes DeepSeek, Mistral, OpenAI, model calls, tokens, and measured cost. It
prints a direct dashboard URL for the execution graph. The contract's approval
result is intentionally not a pass/fail condition—the integration evidence is.

## Model routing

| Task type | Logical role | LiteLLM model | Used for |
|---|---|---|---|
| PDF/image understanding | vision | mistral/mistral-ocr-latest | Scanned PDF OCR, layout-aware page extraction, page evidence |
| Text extraction/transformation | text | deepseek/deepseek-chat | Normalization, classification, metadata, clauses, terms, fallback drafting, obligations |
| Judge/evaluation/approval | judge | openai/gpt-5.4 | Extraction gate, playbook evaluation, severity, fallback gate, final decision |

All LLM and multimodal calls must go through LiteLLM unless LiteLLM cannot expose the capability. Haystack remains the orchestration layer. The official Haystack LiteLLM generator is used for chat calls; the shared role registry calls LiteLLM's OCR endpoint only because the chat generator does not expose document OCR.

Override a role without editing the graph:

~~~bash
export CONTRACT_REVIEW_TEXT_MODEL=anthropic/claude-sonnet-4-5
export CONTRACT_REVIEW_VISION_MODEL=gemini/gemini-2.5-pro
export CONTRACT_REVIEW_JUDGE_MODEL=openai/gpt-5.4
~~~

The defaults deliberately follow the required DeepSeek/Mistral/GPT-5.4 specialization.

## Workflow and branches

~~~text
Input
  -> quality detection
  -> native text? --yes--> native extractor ---------+
                   no----> Mistral OCR via LiteLLM --+
  -> DeepSeek normalize -> classify -> metadata -> clauses -> terms
  -> GPT-5.4 extraction judge
       fail + retries left -> focused DeepSeek re-extraction -> judge
       pass / retry cap reached -> playbook
  -> deviations?
       no  -> straight-through candidate ----------------------+
       yes -> GPT-5.4 risk -> legal / finance / security / business
            -> DeepSeek fallback -> GPT-5.4 fallback judge
                 fail + retries left -> revise fallback --------+
                 pass / retry cap reached ----------------------+
  -> GPT-5.4 final gate
       approved / approved_with_exceptions -> DeepSeek obligations
       manual / rejected / failed -> obligations skipped
  -> strict result assembler
~~~

Final states are approved, approved_with_exceptions, manual_review_required, rejected_by_playbook, and processing_failed.

## Fictional playbook

[vendor_saas.yaml](contract_review_agent/app/playbooks/vendor_saas.yaml) covers term and renewal, termination, liability, indemnity, governing law, assignment, data/security, SLA, and payment. It routes findings to legal, finance, security/privacy, and procurement/business-owner review. These are example business rules, not statements of law.

## Quick start

The locked `uv` workflow is the most reproducible setup:

~~~bash
cd haystack-cuad-contract-review
uv sync --locked --extra dev
cp .env.example .env
uv run python examples/run_demo_suite.py
~~~

The application loads `.env` automatically. The default deterministic mode
does not call model providers and does not require API keys.

This showcase pins Witdem SDK 0.3.0 to the exact [public source commit](https://github.com/ebrahimisoheil/witdem-oss/commit/d15be4572280f5f7f05bda145f2af6903fe79118) that
contains the current Haystack/LiteLLM instrumentation and progressive execution
graph support. The commit pin keeps installation reproducible until the matching
0.3.x SDK release is available from PyPI.

To use a standard virtual environment instead:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
cp .env.example .env
python examples/run_demo_suite.py
~~~

No API keys are required for deterministic examples or tests. Live mode requires the keys for every role reached by a run.

## Environment variables

| Variable | Purpose |
|---|---|
| DEEPSEEK_API_KEY | DeepSeek credential used by LiteLLM |
| MISTRAL_API_KEY | Mistral OCR credential used by LiteLLM |
| OPENAI_API_KEY | OpenAI judge credential used by LiteLLM |
| CONTRACT_REVIEW_MODE | deterministic or live |
| CONTRACT_REVIEW_MODELS_FILE | Alternate declarative role-routing YAML |
| CONTRACT_REVIEW_*_MODEL | Optional TEXT, VISION, or JUDGE model override |
| CONTRACT_REVIEW_MAX_RETRIES | Workflow retry cap; default 2 |
| CONTRACT_REVIEW_TIMEOUT_SECONDS | Provider timeout; default 60 |
| CONTRACT_REVIEW_WITDEM_CONFIG | Optional alternate Witdem contract/configuration file |

Secrets are read from the environment and never serialized into results.

## Run examples

Run all four reproducible paths:

~~~bash
.venv/bin/python examples/run_demo_suite.py
~~~

Review one deterministic fixture-compatible document:

~~~bash
.venv/bin/python examples/run_contract.py path/to/contract.pdf
~~~

Use live providers:

~~~bash
CONTRACT_REVIEW_MODE=live .venv/bin/python -m contract_review_agent.app.main path/to/contract.pdf --mode live
~~~

Run verification:

~~~bash
.venv/bin/pytest
~~~

## Example output

~~~json
{
  "contract_id": "6d4d34eb5f0a3a91",
  "agreement_type": "Vendor SaaS Agreement",
  "parties": {
    "customer": "Acme Example Corporation",
    "vendor": "Risky Cloud Inc."
  },
  "deviations": [
    {
      "clause": "termination",
      "severity": "medium",
      "reason": "Termination for convenience is absent.",
      "evidence": {
        "page": 1,
        "text": "Either party may terminate for cause upon 30 days notice.",
        "clause_label": "termination",
        "extraction_method": "native_pdf_text",
        "confidence": 0.94
      },
      "review_area": "legal"
    }
  ],
  "final_decision": "approved_with_exceptions",
  "review_areas": ["legal", "finance"],
  "outcome": {
    "review_completed": true,
    "evidence_complete": true,
    "playbook_evaluated": true,
    "routing_complete": true,
    "objective_met": true
  }
}
~~~

The actual result also contains all normalized terms, clauses, fallbacks, obligations, errors, and run metrics.

## Observability

Every run emits total and per-stage latency, configured model per stage, LiteLLM token/cost telemetry, retries, full branch path, extraction confidence, deviation and escalation counts, unresolved fields, evidence completeness, and explicit business outcomes.

Deterministic mode reports zero model tokens and cost because it makes no provider calls. It does not present estimates as actual usage.

To inspect runs in Witdem, start a matching 0.3.0 server from a local checkout
of the linked Witdem source commit, then run the examples:

~~~bash
cd /path/to/witdem-oss
docker build -f Dockerfile.witdem -t witdem-local:0.3.0 .
WITDEM_IMAGE=witdem-local:0.3.0 docker compose -f npm/compose.yaml up -d

cd /path/to/haystack-cuad-contract-review
uv run python examples/run_demo_suite.py
~~~

The default [Witdem contract](.witdem/witdem.yaml) exports to
`http://localhost:4318`; the dashboard is available at
`http://localhost:8501`. Tests use an isolated telemetry configuration, so a
test run cannot add synthetic entries to a developer's dashboard.

## Test data

The deterministic suite creates a clean native-text PDF, a raster-only PDF with a page-indexed OCR sidecar that forces the Mistral role, a PDF with legal and finance deviations, and a PDF missing DPA language on the first pass.

These synthetic agreements are small and reproducible. For broader evaluation,
use the [CUAD dataset](https://github.com/TheAtticusProject/cuad), review its
current terms, and preserve original document and label provenance. Downloaded
contracts and generated subsets are deliberately excluded from this repository;
see [data attribution](DATA_ATTRIBUTION.md).

## CUAD ingestion

The CUAD ingestion layer accepts either the official SQuAD-style annotation JSON or a local full CUAD v1 release. It validates every annotated answer offset, records source hashes and attribution, materializes exact contract text, matches original PDFs when present, and produces a strict manifest that can be passed directly to the Haystack workflow.

Download the official annotation archive and create a seeded 20-contract subset:

~~~bash
.venv/bin/cuad-data ingest \
  --download \
  --cache-dir data/cuad-cache \
  --output data/cuad-subset \
  --limit 20 \
  --seed 42
~~~

Ingest a local full release and restrict it to service agreements with positive governing-law labels:

~~~bash
.venv/bin/cuad-data ingest \
  --source /path/to/CUAD_v1 \
  --output data/cuad-service-subset \
  --agreement-type "Service Agreement" \
  --require-category "Governing Law" \
  --limit 20
~~~

Run the selected contracts through the existing workflow:

~~~bash
.venv/bin/cuad-data review data/cuad-subset/manifest.json \
  --mode live \
  --output cuad-runs/cuad-review.json
~~~

The default limit is 20. Limits above 100 require the explicit `--allow-large` flag to prevent accidental high-cost batches. Deterministic mode is useful for testing ingestion, integrity, branching, and audit mechanics; it is not an accuracy benchmark for naturally drafted CUAD contracts. Use live mode for model-quality evaluation.

## Project structure

~~~text
contract_review_agent/
  app/
    components/       visible Haystack business stages
    prompts/          text, vision, and judge prompt assets
    playbooks/        fictional vendor SaaS rules
    config.py         environment settings
    model_registry.py declarative LiteLLM role resolution
    model_routing.yaml
    pipeline.py       Haystack graph and ConditionalRouters
    schemas.py        strict audit schemas
    ingestion/        CUAD download, manifests, labels, and batch handoff
examples/
tests/
.witdem/                 analytics contract and local receiver configuration
~~~

## Business outcome

Success means the contract was reviewed against the supplied fictional playbook, deviations were identified with source evidence, and the correct approval/escalation route was produced. It does not merely mean that an LLM returned JSON. The outcome.objective_met field encodes that distinction.

## Reliability and cost controls

Critical output is Pydantic-validated with forbidden extra fields. Unsupported evidence has zero confidence. Provider errors are captured by stage, structured JSON is rejected when malformed, workflow retries are capped, extracted text is reused, and judges receive focused data rather than the full document. The native branch avoids vision cost; GPT-5.4 is reserved for judgment.

## Limitations

- Deterministic mode validates orchestration and business logic, not provider quality.
- Live-provider behavior, availability, pricing, and output quality vary by account and model version; deterministic results are not provider benchmarks.
- The scan fixture uses a page sidecar to simulate Mistral OCR output while forcing a real raster-only PDF path.
- Complex tables, handwriting, redlines, and exhibits need a broader labeled evaluation set.
- Clause quality varies with drafting and scan quality.
- The fictional rules are intentionally small and are not a production legal playbook.
- Human reviewers remain responsible for legal and commercial decisions.

## Legal disclaimer

> This is a technical demonstration of AI-assisted contract workflow automation using a fictional review playbook. It is not legal advice and should not be used as a substitute for qualified legal review.

## License

Original source code and synthetic fixtures are available under the
[MIT License](LICENSE). CUAD and user-provided contracts remain subject to
their own terms; see [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md).

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). Please report
security issues using the private process in [SECURITY.md](SECURITY.md).
