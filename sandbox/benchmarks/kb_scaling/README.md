# KB Scaling Benchmark

Run from repository root:

```bash
python scripts/benchmark_kb_scaling.py --min-files 2 --max-files 40 --step 2 --queries-file sandbox/prompts/kb_benchmark_queries.txt --output-dir sandbox/benchmarks/kb_scaling
```

The script runs only the `sandbox.kb_agent` file-search path and writes:

- `sandbox/benchmarks/kb_scaling/metrics.csv`
- `sandbox/benchmarks/kb_scaling/latency_vs_files.png`
- `sandbox/benchmarks/kb_scaling/input_tokens_vs_files.png`
- `sandbox/benchmarks/kb_scaling/input_tokens_vs_bytes.png`
- `sandbox/benchmarks/kb_scaling/latency_vs_vector_store_bytes.png`
- `sandbox/benchmarks/kb_scaling/report.md`

Optional:

- Keep created vector stores (debug): add `--keep-vector-stores`
- Override model: add `--model gpt-4o-mini`
