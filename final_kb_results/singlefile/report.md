# KB Growth Benchmark Report — Single-File Growth (Test 1)

## Run config

- model: `gpt-4o-mini`
- step_range: steps `1` to `25`
- step_count: `25`

## Delta metrics (first step → last step)

| Metric | Step 1 | Step 25 | Delta |
| --- | --- | --- | --- |
| ask_elapsed_ms_mean (ms) | 31197.1 | 43152.3 | +11955.2 |
| ask_input_tokens_mean | 8787 | 17471 | +8684 |
| ask_aggregate_throughput (tok/s) | 18.8 | 14.7 | -4.1 |
| preload_elapsed_ms | 7719.2 | 9588.3 | +1869.1 |

## Citation miss count

- Total across all steps: `57` (not excluded from the unfiltered metrics above; responses without file_citation are flagged only).
- Per-query citation data is available in `metrics_per_query.csv` for post-hoc filtering.

## Citation-filtered artifacts

The following filtered plots include only queries that returned a file citation (`has_citation = True`).  Latency and throughput are re-aggregated using a two-stage mean (per-run mean → cross-run mean) with population std error bars, matching the methodology of the unfiltered plots.

- All steps had at least one citation hit across runs.

Artifacts:
- `latency_vs_kb_size_citation_hits_only.png`
- `latency_vs_step_citation_hits_only.png`
- `throughput_vs_kb_size_citation_hits_only.png`

## Methods note

Standard deviation columns (`ask_*_std`) are computed as population standard deviation across runs per step.  Approximate 95% CI: mean ± 2 × std / √R where R is the number of completed runs.

## Artifacts

- `metrics_per_query.csv` — one row per (run, step, query); includes `query_idx`, `query_text`, `has_citation`
- `metrics_per_run.csv` — one row per (run, step)
- `metrics_averaged.csv` — one row per step, averaged across runs
- `latency_vs_kb_size.png`
- `input_tokens_vs_kb_size.png`
- `throughput_vs_kb_size.png`
- `latency_vs_step.png`
- `latency_vs_kb_size_citation_hits_only.png` (when `metrics_per_query.csv` present)
- `latency_vs_step_citation_hits_only.png` (when `metrics_per_query.csv` present)
- `throughput_vs_kb_size_citation_hits_only.png` (when `metrics_per_query.csv` present)
