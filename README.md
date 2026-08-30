# CompositionEvidence

This package contains the code, fixed configuration, selected checkpoint, benchmark
metadata, and per-example evaluation records for this submission. The method
extracts deterministic composition features and injects them into a frozen
vision-language model through either controlled text (Channel A) or learned soft
tokens (Channel B).

Paper title: **CompositionEvidence: Controlling Geometric Evidence for Cinematic
Composition Reasoning with Frozen Vision-Language Models**.

## Contents

| Path | What it is |
|---|---|
| `supplementary/supplementary_material.pdf` | Supplementary material (11 pages): before/after Q&A demos, a counterexample, the exact Channel-A text interface, feature and calibration definitions, Channel-B training details, per-class and corruption results, artifact map, reproducibility summary |
| `src/`, `scripts/`, `configs/` | Extraction, evidence construction, evaluation, and training code with the frozen thresholds |
| `results/` | Per-example records for every condition reported in the paper, plus the validation-selected projector |
| `tests/` | Unit tests (77) |

The two recompute scripts reproduce every number in the paper from the stored
per-example records, with no model download and no network access:

```bash
python scripts/recompute_headline.py
python scripts/recompute_reported.py
```

## Headline result

The included records reproduce the paper's primary Qwen2.5-VL-7B ShotBench
composition result:

| Condition | Correct / N | Accuracy | Wrong-to-right / right-to-wrong vs. C0 |
|---|---:|---:|---:|
| C0 official-prompt baseline | 237 / 479 | 49.5% | -- |
| Channel A: gated evidence after the question | 269 / 479 | 56.2% | 66 / 34 |
| Channel B: four learned soft tokens | 292 / 479 | 61.0% | 100 / 45 |

Recompute the counts and exact two-sided McNemar tests without downloading models:

```bash
python scripts/recompute_headline.py
```

Expected p-values are approximately 0.00179 for Channel A and below 0.00001 for
Channel B.

To verify the main, corruption, ablation, cross-model, cross-dimension, and
cross-benchmark numbers from all shipped per-example records, run:

```bash
python scripts/recompute_reported.py
python scripts/audit_release.py
```

## Scientific provenance

The final numeric gate values in `configs/default.yaml` and
`configs/calibration.json` were fitted using ShotQA training data only. However,
the decision to introduce F4/F5 calibration and the final rendering design was
made after first-pass diagnostics on ShotBench. Therefore, the values are
training-fitted, but the overall system design is adaptive to the target
benchmark. The shipped ShotBench records should not be described as an
independent confirmatory test or as a prospectively registered analysis.

The sensitivity sweeps in `results/ablation/` are diagnostic analyses. They are
included to make the design history and robustness checks inspectable, not to
retroactively claim independent model selection.

## Installation

Python 3.11 or later is required. The base environment supports metric
recomputation, feature tests, and plotting:

```bash
python -m pip install -e ".[dev]"
pytest
```

Model inference and training require the optional model stack:

```bash
python -m pip install -e ".[server]"
source scripts/env.sh
```

Official commercial-model API reruns require the provider SDKs:

```bash
python -m pip install -e ".[api]"
source scripts/env.sh
```

`scripts/env.sh` places package and model caches inside this project. Override
TensorBoard output with `CE_TF_LOG_DIR`; otherwise it is written to `logs/tf`.
No credentials are stored here.

## Official provider APIs

The five commercial-model evaluations use official provider APIs, recorded in
`configs/models.yaml` with an evaluation month of July 2026:

| Models | Provider interface | Credential variables |
|---|---|---|
| gpt-5.4 | OpenAI Responses API | `OPENAI_API_KEY` |
| gemini-3.1-pro | Google Gemini API | `GEMINI_API_KEY` |
| claude-opus-4-8 | Anthropic Messages API | `ANTHROPIC_API_KEY` |
| qwen-vl-max, qwen3-vl-plus | Alibaba Cloud Model Studio | `DASHSCOPE_API_KEY`, `DASHSCOPE_BASE_URL` |

OpenAI, Google, and Anthropic calls use their official Python SDKs without a
configurable base URL. Alibaba Model Studio officially exposes an
OpenAI-compatible multimodal interface whose URL depends on account region and
workspace; the code accepts only HTTPS hosts under `aliyuncs.com`. Arbitrary
third-party endpoints are intentionally unsupported.

Provider aliases can be updated over time. Before a rerun, confirm that the
listed model identifiers still resolve to the intended provider versions. The
included per-example records, rather than a future rerun of a moving alias, are
the auditable artifact for the scores in the paper.

## Data

The package includes ShotBench annotations and metadata but not copyrighted
images or videos. ShotQA training media and CineTechBench media are also not
redistributed. See `data/README.md` for the expected layout.

```bash
python scripts/download_data.py --which shotbench
python scripts/download_data.py --which shotqa
python scripts/audit_benchmark.py
```

The download commands use the public dataset identifiers recorded in
`scripts/download_data.py`. CineTechBench annotations can be downloaded there;
its media must be obtained using the links supplied by that benchmark.

## Reproduction workflows

Channel A:

```bash
python scripts/calibrate_thresholds.py
python scripts/fit_gates.py
python scripts/run_extract.py
python scripts/build_evidence.py
python scripts/run_final.py --model qwen2_5_vl_7b --conditions C0,FINAL
```

Channel B:

```bash
python scripts/extract_shotqa_features.py
python scripts/train_projector.py --out-dir results/projector
python scripts/eval_c4.py --ckpt results/projector/best.pt
```

The validation-selected four-token projector checkpoint is included at
`results/projector/best.pt`. Pass `--device` to the projector scripts when the
default `cuda:0` is not appropriate.

Core ablations and figures:

```bash
python scripts/run_ablation.py --mode groups --model qwen2_5_vl_7b
python scripts/run_ablation.py --mode perturb_final --model qwen2_5_vl_7b
python scripts/run_ablation.py --mode thresholds --model qwen2_5_vl_7b
python scripts/make_figures.py --model qwen2_5_vl_7b
```

All evaluation loops append records and skip completed UIDs, so interrupted runs
can resume. Remove or relocate a specific output file before intentionally
starting that condition from scratch.

## Implementation notes

- Dominant-subject selection combines detector confidence, normalized box area,
  and a center prior, with a configurable person-class boost. Near-full-frame
  boxes are rejected because they carry little positional information.
- Extractor caches are keyed by a SHA-1 digest of media bytes, rather than by an
  absolute filename, so moving the project does not invalidate cache identity.
- The 24-dimensional evidence vector contains continuous measurements and
  presence flags for missing observations.
- Channel A renders factual measurements only. A lexical guard rejects evidence
  whose word roots overlap answer labels. The final recipe puts gated evidence
  after the official question and options.
- Symmetry and visual-mass thresholds are computed with Youden's J on ShotQA
  training data. Percentile context is derived from the stored training
  distribution. Stronger precision-oriented gates are retained as an ablation.
- Channel B maps the 24-dimensional vector through a two-layer MLP to four
  embedding vectors. A forward hook replaces four placeholder embeddings; all
  vision-language model weights remain frozen.
- Evaluation records store UIDs, parsed answers, raw model responses, ground
  truth, and correctness, enabling paired tests without rerunning inference.

## File map

### Configuration and data

| Path | Purpose |
|---|---|
| `pyproject.toml` | Package metadata and base, model, vLLM, and test dependencies. |
| `configs/default.yaml` | Seeds, paths, thresholds, extractor settings, feature set, and decoding controls. |
| `configs/prompts.yaml` | Official question formatting, evidence templates, and answer-label lexical guard. |
| `configs/models.yaml` | Local checkpoints plus official API providers, model identifiers, evaluation month, and credential-variable names. |
| `configs/calibration.json` | Frozen ShotQA training distributions, fitted thresholds, and evidence gates. |
| `configs/calibration_rows.json` | Labeled training measurements used by the gate-fitting script. |
| `data/README.md` | Dataset acquisition, redistribution, and directory-layout notes. |
| `data/shotbench/test.tsv` | Public ShotBench annotations. |
| `data/shotbench/meta.parquet` | Derived benchmark metadata used for audits. |
| `data/shotbench/sc_labels.json` | Composition answer vocabulary used by leakage checks. |

### Entry points

| Path | Purpose |
|---|---|
| `scripts/env.sh` | Keeps caches local and exposes optional endpoint configuration. |
| `scripts/check_env.py` | Reports dependency versions and runs a synthetic CPU smoke test. |
| `scripts/download_data.py` | Downloads public ShotBench, ShotQA, and CineTechBench artifacts. |
| `scripts/audit_benchmark.py` | Audits ShotBench categories, labels, media types, and optional resolutions. |
| `scripts/run_extract.py` | Runs frozen box, depth, and line extractors over ShotBench media. |
| `scripts/build_evidence.py` | Converts cached observations to features and controlled evidence text. |
| `scripts/calibrate_thresholds.py` | Fits F4/F5 thresholds and percentile distributions on ShotQA training data. |
| `scripts/fit_gates.py` | Fits precision- and coverage-controlled evidence gates. |
| `scripts/run_eval.py` | Runs C0--C3 conditions and produces summary tables. |
| `scripts/run_final.py` | Runs C0 and the final post-question gated-text condition. |
| `scripts/run_ablation.py` | Runs feature-group, perturbation, position, style, gate, and threshold ablations. |
| `scripts/run_rule_baseline.py` | Evaluates the deterministic feature-to-label rule baseline. |
| `scripts/extract_shotqa_features.py` | Builds training vectors and labels for the soft-token projector. |
| `scripts/train_projector.py` | Trains the frozen-VLM soft-token projector with a held-out validation split. |
| `scripts/eval_c4.py` | Evaluates a trained projector on ShotBench composition questions. |
| `scripts/run_ctb.py` | Evaluates the final controlled-text recipe on CineTechBench composition items. |
| `scripts/make_figures.py` | Generates confusion matrices, geometric buckets, and case-study UID lists. |
| `scripts/make_nine_model_fig.py` | Generates the cross-model summary figure from recorded scores. |
| `scripts/audit_release.py` | Checks archive hygiene, configs, checkpoint structure, JSONL schemas, and duplicate UIDs. |
| `scripts/recompute_headline.py` | Recomputes and checks the three headline result rows. |
| `scripts/recompute_reported.py` | Recomputes and checks reported main, corruption, cross-model, cross-dimension, and transfer results. |

### Source modules

| Path | Purpose |
|---|---|
| `src/utils.py` | Configuration, seeding, content hashes, JSON conversion, and resumable JSONL utilities. |
| `src/extractors/base.py` | Common cached-extractor interface. |
| `src/extractors/grounding_dino.py` | GroundingDINO subject-box extraction. |
| `src/extractors/depth.py` | Depth Anything V2 relative-depth extraction. |
| `src/extractors/lines.py` | LSD line extraction with a Hough fallback. |
| `src/extractors/saliency.py` | Pseudo-saliency from boxes and relative depth. |
| `src/extractors/frames.py` | Uniform and middle-frame video sampling. |
| `src/features/subject_selection.py` | Deterministic dominant-subject selection. |
| `src/features/geometry.py` | Centroid, thirds, area, and frame-edge geometry. |
| `src/features/balance.py` | Left/right visual-mass measurement. |
| `src/features/symmetry.py` | Horizontal and vertical mirror SSIM. |
| `src/features/depth_layers.py` | Foreground, middle, and background depth summaries. |
| `src/features/lines.py` | Leading-line and horizon-related measurements. |
| `src/features/headroom.py` | Face-top and head-space measurements. |
| `src/features/compute.py` | Assembles the complete structured feature record. |
| `src/injection/prompts.py` | Builds C0--C3 prompts and controls evidence position. |
| `src/injection/render.py` | Renders numeric evidence, applies gates, and enforces lexical safety. |
| `src/injection/perturb.py` | Applies controlled feature perturbations for causal diagnostics. |
| `src/injection/soft_token.py` | Featurizes records and builds the two-layer soft-token projector. |
| `src/eval/datasets.py` | ShotBench schema and subset loaders. |
| `src/eval/backends.py` | Local Qwen/Transformers inference plus explicit OpenAI, Google, Anthropic, and Alibaba Model Studio backends. |
| `src/eval/parsing.py` | Extracts option letters from model responses. |
| `src/eval/runner.py` | Resumable paired evaluation loop and media resolution. |
| `src/eval/metrics.py` | Accuracy, paired flips, and exact McNemar tests. |
| `src/eval/rule_baseline.py` | Maps structured evidence directly to answer options. |
| `src/eval/tblog.py` | Text and optional TensorBoard metric logging. |
| `src/viz/figures.py` | Confusion-matrix and geometric-bucket helpers. |
| `src/viz/overlays.py` | Draws detector and geometry overlays for qualitative examples. |
| `src/**/__init__.py` | Package markers. |

### Tests and shipped artifacts

| Path | Purpose |
|---|---|
| `tests/conftest.py` | Shared synthetic feature and prompt fixtures. |
| `tests/test_calibration.py` | Threshold and gate-fitting unit tests. |
| `tests/test_eval_units.py` | Response parsing and paired-metric tests. |
| `tests/test_official_api_config.py` | Official-provider routing and Alibaba endpoint-domain checks. |
| `tests/test_geometry.py` | Geometry feature tests. |
| `tests/test_lines_headroom.py` | Line, horizon, and head-space tests. |
| `tests/test_render_perturb.py` | Evidence rendering, lexical safety, gating, and perturbation tests. |
| `tests/test_rule_baseline.py` | Direct rule-baseline tests. |
| `tests/test_subject_selection.py` | Dominant-subject scoring and rejection tests. |
| `tests/test_symmetry_balance_depth.py` | Mirror, visual-mass, and depth-layer tests. |
| `results/eval/*.jsonl` | Per-example main, cross-model, harmlessness, and cross-benchmark records. |
| `results/ablation/*.jsonl` | Per-example ablation and perturbation records. |
| `results/evidence/shotbench_sc.jsonl` | Structured features and controlled text for the SC subset. |
| `results/projector/best.pt` | Validation-selected four-token projector weights. |

## Scope and limitations

The three frozen perception models and the evaluated VLM checkpoints must be
downloaded separately. Exact GPU memory and runtime depend on hardware and model
backend. API-based reruns depend on access to the corresponding official
provider model versions and may change if a provider updates a moving model
alias. The included per-example records are the stable artifact for auditing the
reported numbers.
