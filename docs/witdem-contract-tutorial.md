# Why this repository has a Witdem YAML contract

The canonical file in this repository is
[`.witdem/witdem.yaml`](../.witdem/witdem.yaml). Witdem also accepts a custom
`.yml` filename when it is passed explicitly, but `.witdem/witdem.yaml` is the
convention discovered automatically by the SDK.

The short version is:

> Haystack and LiteLLM telemetry tells Witdem **what ran**. The Witdem YAML
> contract tells Witdem **what the returned result means**.

A completed pipeline is not necessarily a successful contract review. It may
have returned incomplete evidence, skipped the playbook, chosen an escalation,
or failed to produce a usable result. The YAML file gives those application
facts stable names and mappings without putting analytics logic into the
Haystack graph.

## The result the contract evaluates

Witdem instruments the Haystack `Pipeline` in
[`pipeline.py`](../contract_review_agent/app/pipeline.py). Haystack returns the
final component output in this shape:

```json
{
  "result_assembler": {
    "result": {
      "contract_id": "…",
      "final_decision": "manual_review_required",
      "decision_explanation": "…",
      "outcome": {
        "review_completed": true,
        "playbook_evaluated": true,
        "routing_complete": true,
        "objective_met": true
      },
      "metrics": {
        "evidence_completeness": 1.0,
        "extraction_confidence": 0.9,
        "retries": 2,
        "deviation_count": 4,
        "escalation_count": 2,
        "total_input_tokens": 8000,
        "total_output_tokens": 1500,
        "estimated_cost_usd": 0.03
      }
    }
  }
}
```

Paths such as `$.result_assembler.result.final_decision` are declarative reads
from this returned value. Witdem does not ask another model to guess what the
fields mean.

## Contract anatomy

### 1. Identify the application

```yaml
version: 1

service:
  name: haystack-cuad-contract-review
  description: Haystack and LiteLLM workflow for contract review against a fictional playbook.
  runtime: haystack
```

Why:

- `version` makes the configuration grammar explicit.
- `service.name` gives runs a stable system identity for grouping and comparison.
- `runtime: haystack` records the orchestration framework being instrumented.
- The description explains the example's scope and avoids presenting the
  fictional playbook as legal advice.

### 2. Configure telemetry and privacy

```yaml
telemetry:
  endpoint: http://localhost:4318
  mode: auto
  capture_content: false
```

Why:

- The endpoint sends telemetry to the local Witdem receiver.
- `auto` installs an exporter when the process does not already have a usable
  telemetry provider.
- `capture_content: false` keeps contract text, prompts, and model responses out
  of telemetry by default. Witdem still receives structural execution evidence,
  declared business facts, usage, and measurements.

Use `WITDEM_ENDPOINT` when a deployment needs a different receiver. Secrets and
provider keys belong in environment variables, never in this YAML file.

### 3. Select the business contract

```yaml
default_contract: contract_review

contracts:
  contract_review:
    mode: expression
  cuad_contract_review:
    mode: reported
```

Why:

- A service can define more than one business contract, so the default is
  explicit.
- `expression` means Witdem evaluates paths and expressions against Haystack's
  final returned value.
- `reported` is used by the CUAD batch path because the application computes
  held-out ground-truth scores after the prediction is complete.

Ordinary reviews use the default expression contract because the application
already returns a strict Pydantic result. CUAD reviews select the reported
contract through the Haystack integration's result reporter. Separating the two
prevents ordinary reviews from receiving fake zero scores when no ground truth
exists.

### 4. Record the application outcome

```yaml
application_outcome:
  status: $.result_assembler.result.final_decision
```

Why:

Runtime status and business status are different. The execution can complete
successfully while the review disposition is `manual_review_required` or
`rejected_by_playbook`. This mapping makes the domain result visible instead of
reducing every technically successful run to `completed`.

### 5. Define a valid returned artifact

```yaml
artifact:
  name: Contract review result
  description: The structured review returned by the Haystack workflow.
  valid:
    all:
      - non_empty: $.result_assembler.result.contract_id
      - $.result_assembler.result.outcome.review_completed
```

Why:

The useful artifact is the structured contract review—not an individual model
response. It is valid only when it has a stable contract identifier and the
review reached a completed business state. An escalation can therefore be a
valid artifact even though it is not an automatic approval.

### 6. Name the business decision

```yaml
decision:
  name: Review disposition
  observed: $.result_assembler.result.final_decision
  reason: $.result_assembler.result.decision_explanation
  outcomes:
    approved: Approved without exceptions.
    approved_with_exceptions: Approved with documented exceptions.
    manual_review_required: Escalated for human review.
    rejected_by_playbook: Rejected by the fictional playbook.
    processing_failed: The workflow did not complete the review.
```

Why:

- `observed` preserves the actual decision returned by the application.
- `reason` keeps the human-readable explanation connected to that decision.
- `outcomes` translates machine labels into dashboard language.
- There is no `expected` value because this showcase does not contain an
  independent ground-truth disposition for every live contract. Witdem should
  not claim decision correctness without one.

### 7. Define product success separately

```yaml
product_goal:
  name: Evidence-backed contract review completed
  description: Apply the fictional playbook and return an evidence-backed approval or escalation route.
  achieved: $.result_assembler.result.outcome.objective_met
```

Why:

The product goal is not “automatically approve the contract.” The useful job is
to return an evidence-backed **approval or escalation route**. The application
sets `objective_met` only when:

1. the review completed;
2. the playbook was evaluated;
3. routing completed; and
4. evidence completeness is at least `0.8`.

That is why a run may correctly show:

```text
Review disposition: Manual review required
Product goal: Achieved
```

The system succeeded at identifying and routing work that still requires human
judgment.

If the product requirement changed to automatic approval only, the contract
could make that explicit without changing the Haystack graph:

```yaml
product_goal:
  name: Contract automatically approved
  achieved:
    all:
      - $.result_assembler.result.outcome.objective_met
      - equals:
          - $.result_assembler.result.final_decision
          - approved
```

### 8. Add quality evaluations

```yaml
evaluations:
  - name: Evidence completeness
    score: $.result_assembler.result.metrics.evidence_completeness
    target: 0.8
    direction: higher_is_better
    unit: ratio
  - name: Extraction confidence
    score: $.result_assembler.result.metrics.extraction_confidence
    target: 0.7
    direction: higher_is_better
    unit: ratio
```

Why:

Evaluations explain result quality and place the observed scores beside declared
targets. They power the participant-quality analysis shown in the Witdem
dashboard. Evaluations are evidence; they do not silently redefine product-goal
success. In this application, evidence completeness also affects
`objective_met` because the application explicitly includes that threshold in
its own outcome calculation.

The reported CUAD contract declares a separate evaluation glossary:

```yaml
cuad_contract_review:
  mode: reported
  evaluations:
    category_f1:
      name: CUAD category F1
      target: 0.7
      direction: higher_is_better
      unit: ratio
    span_token_f1:
      name: CUAD evidence span token F1
      target: 0.5
      direction: higher_is_better
      unit: ratio
    negative_label_accuracy:
      name: CUAD negative-label accuracy
      target: 0.8
      direction: higher_is_better
      unit: ratio
```

The application supplies these values only when a review comes from a CUAD
manifest. Ground-truth labels stay in a closure owned by the post-run result
reporter and are evaluated after all model stages have completed. They never
become Haystack input or enter the context seen by extraction, judge, or retry
components. The evaluator compares the finished prediction with seven
defensibly mapped CUAD categories and reports category F1, evidence-span token
F1, and negative-label accuracy to Witdem on the same execution. A score with
no valid denominator is omitted rather than invented.

These held-out scores do not drive a retry on the same contract. Doing so would
leak the answer. They support offline model-routing, prompt, and workflow
experiments across a batch. The application does use the three declared targets
to report Witdem evidence sufficiency. Product-goal achievement can therefore
remain true while assurance needs attention, preserving the distinction between
completing the business route and proving its held-out quality.

### 9. Keep diagnostics separate from success

```yaml
metrics:
  - name: Workflow retries
    value: $.result_assembler.result.metrics.retries
    unit: retries
  - name: Deviations found
    value: $.result_assembler.result.metrics.deviation_count
    unit: deviations
  - name: Escalations
    value: $.result_assembler.result.metrics.escalation_count
    unit: review_areas
```

The contract also records input tokens, output tokens, and application-reported
model cost. These values help explain behavior and economics, but they do not
make a review successful or unsuccessful. Keeping measurements separate from
the goal prevents a cheap or fast failure from looking like a good product
outcome.

The reported CUAD contract separately records evaluated categories, true and
false positives, true and false negatives, evaluated spans, and workflow
retries. These diagnostics make an aggregate F1 score investigable rather than
opaque.

Witdem can additionally derive provider/model usage and measured cost from the
observed LiteLLM calls. Application-reported cost remains a separate declared
measurement rather than overwriting provider evidence.

### 10. Add dimensions for investigation

```yaml
attributes:
  contract_id: $.result_assembler.result.contract_id
  agreement_type: $.result_assembler.result.agreement_type
  final_decision: $.result_assembler.result.final_decision
```

Why:

Attributes make runs filterable and comparable without capturing the contract
body. They let an operator investigate a contract, agreement category, or
decision population while preserving the default content policy.

## How the layers fit together

```text
Haystack instrumentation  -> components, branches, joins, retries
LiteLLM instrumentation   -> provider, model, tokens, latency, cost
Witdem YAML contract      -> artifact, decision, product goal, evaluations
Witdem dashboard          -> run graph plus cross-run business intelligence
```

The contract is deliberately declarative. Model routing can change from
DeepSeek to another text model, for example, without changing what counts as a
successful contract review. Likewise, the business definition can evolve
without scattering analytics conditions throughout the orchestration code.

## Validate changes

Run validation from the repository root:

```bash
uv run witdem-sdk validate --config .witdem/witdem.yaml
```

Then run the deterministic tests:

```bash
uv run --extra dev pytest
```

For an end-to-end telemetry check, start Witdem and run the live verifier:

```bash
npx -y witdem@stable-0-1 up
uv run contract-review-showcase output/showcase/scanned-vendor-saas.pdf
```

The verifier requires Witdem to observe a terminal Haystack execution with
DeepSeek, Mistral, OpenAI, model calls, tokens, and measured cost.

## Design checklist

When changing this contract, confirm that:

- the product goal describes user value rather than runtime completion;
- the artifact is the final useful object, not an intermediate model response;
- decision correctness is declared only when independent ground truth exists;
- result paths still match the real Haystack return shape;
- evaluations have meaningful targets and directions;
- diagnostic measurements are not accidentally treated as success criteria;
- content capture remains an explicit privacy decision; and
- `witdem-sdk validate` passes before committing.

For the general contract grammar and more examples, see the
[Witdem YAML contract tutorial](https://github.com/ebrahimisoheil/witdem-oss/blob/main/docs/contract-tutorial.md).
