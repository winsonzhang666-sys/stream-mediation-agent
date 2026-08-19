# Streaming Dispute-Mediation Assistant · Real-Time Case-State Tracking over Long Conversations

**English** | [简体中文](README.zh-CN.md)

**While a mediation conversation is still in progress**, the model incrementally maintains and emits the structured state of the case — each party's **claims / established facts / points of dispute / evidence / settlement proposals** — helping the mediator clarify the case as they talk, instead of summarizing everything once the session is over.

---

## 1. What this project solves

During a mediation session, a community mediator has to keep track of five kinds of information as the dialogue unfolds: each party's **claims**, the **facts** already established, the **points of dispute**, the **evidence**, and any viable **settlement proposal** — and work from that.

The project originally took the "after mediation ends, generate a case summary and structured fields in one pass from the full transcript" approach. As the work progressed it became clear that this **post-hoc summarization** mode does not fit the real scenario: what mediators actually need is the case state **continuously updated while the mediation is under way**, not a summary report handed to them afterwards.

The system was therefore upgraded into a **streaming state-tracking framework for long conversations** — every few turns of dialogue it emits "the full state of the case as of right now", so the mediator can watch the state change as they talk. This is also the single biggest challenge in the project.

Compared with one-shot generation, streaming generation has to solve two additional problems:

- **Abstention when information is insufficient**: facts, amounts and evidence that never appeared in the dialogue must be left blank — the model must never fill them in. Better empty than invented.
- **Maintaining state rather than rewriting it every turn**: after each round the model must **accurately update only what changed** on top of what is already there, while keeping the state consistent across the whole session, instead of regenerating everything.

This means the model has to not only understand the current input, but also keep the state continuous and self-consistent throughout the entire mediation.

---

## 2. Input / Output

The model **works incrementally, turn by turn**: each step takes 「**the accumulated full state of the five elements + this round's new dialogue**」 and emits only 「**this round's changes (diff)**」. The complete state is rolled forward and fed back by the program between steps.

**Input** (what is fed to the model at each step):
- the **new dialogue** for this round (whatever was said since the last snapshot, with turn number, speaker and role);
- the **roster** (who is the applicant / respondent / mediator, mapped to real names);
- the **case state accumulated so far** (all five elements, in full).

**Output**: a **snapshot of the case state** as of this step — the five elements, each carrying structured sub-fields:

| Element | Key sub-fields (selected) |
|---|---|
| Claims | id, claim content, current status (newly raised / under negotiation / accepted / rejected …), source turns |
| Established facts | id, fact description, current status (awaiting one side's response / confirmed by both / disputed …), each party's stance (who admits / denies / has not responded) |
| Points of dispute | id, disputed question, current status, linked claims / facts, what information is still missing |
| Evidence | id, evidence content, submission status (mentioned but not uploaded / submitted / verified …) |
| Settlement proposal | proposal content, items pending confirmation, each party's stance, current status, plus **strategy scripts** — step-by-step talking-point suggestions organized along four dimensions, **empathy / interest / reason / law**, together with "what not to say, what to watch for, and when to switch strategy" |

> For the complete field list, the value vocabularies and a runnable synthetic example, see [`docs/数据格式_输入输出_纯格式版.md`](docs/数据格式_输入输出_纯格式版.md).

---

## 3. Architecture

The core of streaming state tracking = **the program maintains the full state + the model writes only the changes**.

### Core: incremental (diff) streaming generation

The model is *not* asked to re-emit the whole state every round (which brings repeated generation, loss of history, and even structural deletion of already-confirmed content). Instead:

```
Program side  maintains the 【complete accumulated state】  —— deterministic accumulation, no reliance on the model
Model side    generates only 【this round's changes】       —— additions / updates / a new settlement proposal
Unchanged items → carried into the next snapshot automatically by the program
```

- the state can be **continuously accumulated and updated**;
- "deleting history on its own initiative" is **structurally impossible** — the model has no "delete" action, only add or update;
- the model's input is bounded at each step, and its output is small and clean.

### What the program side (wrapper layer) does

Turning the model's "diff for this step" into "a usable current full state" is the **wrapper layer's** job — deterministic throughout, with no reliance on the model:

1. **Accumulate**: merge this step's diff into the maintained full state — appending added items, editing updated items in place, leaving unchanged items as they are.
2. **Dedup / empty-update filtering**: occasionally the model re-emits items that carry no substantive change; the wrapper compares old and new values and **discards anything whose value did not change**, so it can neither pollute the state nor produce meaningless repeated rewrites.
3. **Consistency regression**: after each accumulation step, the structural invariants are checked (has a status word gone out of vocabulary, are the ids self-consistent, is a future turn being referenced, has a still-valid item been dropped) — if the check fails, the step is blocked.

In one sentence: **the model handles understanding and change, the program handles state and discipline** — the model need not memorize the full state and has no way to damage history, and structural correctness is backstopped by the program.

---

## 4. Evaluation metrics for the data and the model

Evaluation has two layers: first measure how clean the **data** is (pre-training check + pre-delivery self-inspection), then measure how good the **model output** is.

### 4.1 Data-quality metrics (training only starts once these pass)

| Metric | Definition | Passing line |
|---|---|---|
| No-fabrication rate (grounding) | every element can be traced to a source in the dialogue | spot check ≈ 100% / fabrication ≈ 0 |
| Attribution accuracy | every stance is attributed to whoever actually said it | spot check ≈ 100% |
| Status-vocabulary closure | state-machine field values ∈ the closed vocabulary | 100% |
| Future-information leakage | every source turn ≤ this snapshot's cutoff | 0 |
| Unauthorized-deletion rate | a still-valid item disappears in a later snapshot | 0 |
| Id self-consistency | one matter one id, no id splitting, prefixes continue in order | 100% |
| Arithmetic self-consistency | sums, installments and discounts add up before and after | 0 errors |

> The structural ones (vocabulary / leakage / deletion / ids) can be caught dead by rules → hard gates, must be driven to zero; the semantic ones (fabrication / attribution) are the hard part → drive spot checks toward 0 and fix on discovery.

### 4.2 Model-output metrics

| Dimension | Metric | How it is judged |
|---|---|---|
| Structural compliance | closed status vocabulary, self-consistent ids, no leakage, no unauthorized deletion | rule-checkable automatically, should be ≈100% / 0 |
| State tracking | state agreement rate (model state vs gold), coverage (every item that should be carried is carried) | item-by-item comparison |
| Grounding | no-fabrication rate (no information from outside the dialogue in the output) | trace back to the source |
| Proposal / script quality | good rate / bad rate (do the proposal and scripts hold to this case's facts, is there empty boilerplate) | multi-referee scoring |
| Attribution accuracy | stances not misattributed, no collapsed keys | trace back to the source |
| Generation stability | no repetition degeneration (the diff does not fall into a loop) | rules + human review |

> **Judging method (LLM-as-judge · multi-referee cross review)**: the referee panel is drawn from different model families — **Opus 4.7 / 4.8, Fable 5, Opus 5, GPT 5.6** and others — and each round requires **three referees scoring independently with a ≥2-vote consensus** before a finding is accepted; a single referee over-reports easily, whereas cross-family review lowers the false-positive rate.

### 4.3 One conclusion

**Format is not the bottleneck, semantic discipline is**: a well-formed diff can be produced by the bare base model on the system prompt alone; what fine-tuning actually buys is the **semantic discipline** of "don't fabricate, don't delete, don't collapse attribution, keep status words converged".

---

## 5. Deployment / usage

There are two ways to use this repo: **take the adapter only** (hot-mount the LoRA on your own base model and implement the diff→full-state accumulation yourself), or **use the wrapper layer shipped with the repo** (`serving/`, i.e. the production implementation of the "program side" in §3).

### Prerequisites (cannot be bundled in a public repo — bring your own)

1. **Base model `Qwen3.6-35B-A3B`** (MoE, architecture `qwen3_5_moe`, ~35B total / ~3B active params): not publicly downloadable; the adapter can only be mounted on this base, and your transformers / vLLM must support the architecture.
2. The adapter weights `adapters/v4g/adapter_model.safetensors` (~172 MiB) are distributed via **Git LFS** — after cloning, confirm that `git lfs` is installed and that what you pulled is the real weights rather than an LFS pointer file.
3. The wrapper layer depends on `fastapi` / `uvicorn` / `requests`.

**Bring up the full wrapper layer** (the production implementation of the "program side" in §3: vLLM backend + FastAPI normalization service):

```bash
# backend 8000 (base + LoRA) + wrapper 8001
MEDIATION_API_KEY=<your-token> BASE_MODEL=/path/to/Qwen3.6-35B-A3B TP=4 \
  bash serving/start_lora_v4g.sh
```

Contract: `POST /api/mediation/generate` {当事人, 新对话, 当前累积状态?} → {code, message, data, diff}.

---

## 6. Repository structure

This repo is centered on **adapter + wrapper layer + inference demo + docs**; the data, and the engineering scripts for data generation / cleaning / QC / gating / evaluation / training, are not published with it (they involve internal recipes and real data). The system prompts *are* published (`prompts/`, mandatory for inference).

```
stream-mediation-agent/
├── README.md                          ← this file (English)
├── README.zh-CN.md                    ← Chinese version
├── prompts/
│   ├── system_cprime_procfilter_v4.txt ← wrapper-layer default
│   └── system_v4g.txt                 ← the one used to train the v4g adapter (train==serve, single source of truth)
├── adapters/
│   └── v4g/                           ← v4g LoRA adapter (not merged, hot-mounted for serving)
│       ├── adapter_config.json
│       ├── adapter_model.safetensors  ← ~172 MiB, distributed via Git LFS
│       └── README.md                  ← provenance (base model / hyper-params / best ckpt)
├── serve/
│   ├── serve_vllm_lora.sh             ← bare vLLM serving base + LoRA (parameterized)
│   └── demo_infer.py                  ← minimal inference demo (stdlib only, synthetic case built in)
├── serving/                           ← production wrapper layer (the full implementation of §3 "program side")
│   ├── app_stream_v4g.py              ← FastAPI service: POST /api/mediation/generate
│   ├── conv_prompts_stream_v4g.py     ← system/user assembly; SYS_FILE switches within prompts/
│   ├── normalize_diff_v4g.py          ← diff normalization: accumulate / drop empty updates / anchor descriptions / hard-rule filtering / gray-zone referee
│   └── start_lora_v4g.sh              ← one-shot launch of backend vLLM + wrapper
└── docs/
    ├── 数据格式_输入输出_纯格式版.md    ← field schema + value vocabularies + feedback loop + synthetic example
    ├── diff走查示例.md                 ← a step-by-step diff walkthrough of one case
    └── 数据方_质量要求与检修规范.md     ← the data provider's view: hard quality constraints + per-dimension repair
```

> Wrapper-layer credentials always go through environment variables (`MEDIATION_API_KEY` for inbound auth, `BACKEND_API_KEY` for calling the backend); there is no hardcoded secret anywhere in the repo, and `BASE_MODEL` must be given explicitly. The request-retention module `reqlog` writes real conversations to disk and is **not published with the repo** — `app_stream_v4g.py` degrades gracefully in its absence, with no effect on the main flow.

---

## 7. Document index (docs/)

The documents are in Chinese.

- [`docs/数据格式_输入输出_纯格式版.md`](docs/数据格式_输入输出_纯格式版.md) —— field list, value vocabularies, runnable synthetic example.
- [`docs/diff走查示例.md`](docs/diff走查示例.md) —— a worked example of what one step's diff actually looks like.
- [`docs/数据方_质量要求与检修规范.md`](docs/数据方_质量要求与检修规范.md) —— the data provider's view: the hard quality constraints for getting it right first time + the per-dimension inspection criteria / thresholds / fixes for afterwards.
