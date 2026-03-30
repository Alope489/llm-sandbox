# Replication Agents Benchmark

Run from repository root:

```bash
python scripts/benchmark_replication_agents.py --agents extractor,processor,kb,simulation --replication-factors 1,2,4,8,16 --repetitions 3 --workers 8 --output-dir sandbox/benchmarks/replication_agents
```

The benchmark compares `linear` vs `parallel` coordinator execution on replicated workloads per routed agent and writes:

- `sandbox/benchmarks/replication_agents/metrics.csv`
- `sandbox/benchmarks/replication_agents/speedup_vs_replication.png`
- `sandbox/benchmarks/replication_agents/elapsed_ms_vs_replication_linear_parallel.png`
- `sandbox/benchmarks/replication_agents/throughput_vs_replication.png`
- `sandbox/benchmarks/replication_agents/report.md`

Optional:

- Benchmark only selected agents: `--agents extractor,processor`
- Change replication sweep: `--replication-factors 1,3,6,12`
- Change run count: `--repetitions 5`
- Select KB scale: `--kb-scale small`
- Select simulation mode: `--sim-mode regular`
