# KB Scaling Benchmark Report

## Run config

- model: `gpt-4o-mini`
- query_count_per_step: `6`
- step_count: `10`
- min_files: `4`
- max_files: `40`

## Artifacts

- `metrics.csv`
- `latency_vs_files.png`
- `input_tokens_vs_files.png`
- `input_tokens_vs_bytes.png`
- `latency_vs_vector_store_bytes.png`

## Observations

- Mean ask latency changes from `9015.897` ms to `35548.212` ms across the sweep.
- Mean ask input tokens change from `1730.333` to `10333.333`.
- Local KB bytes change from `2116` to `98017`.

## Notes

- Input token values come from provider usage telemetry and include full context assembled for the request.