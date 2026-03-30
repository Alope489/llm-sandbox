# Replication Benefit Benchmark Report

## Scope

- agents: `extractor, kb, processor, simulation`
- replication_factors: `1, 2, 4`
- repetitions_per_point: `1`

## Ranking by replication benefit

- `1. processor` mean_speedup_factor_gt1=`2.241778` max_speedup_factor_gt1=`2.245868`
- `2. kb` mean_speedup_factor_gt1=`1.818539` max_speedup_factor_gt1=`2.318712`
- `3. extractor` mean_speedup_factor_gt1=`1.307749` max_speedup_factor_gt1=`1.50764`
- `4. simulation` mean_speedup_factor_gt1=`1.298435` max_speedup_factor_gt1=`1.323973`

## Artifacts

- `metrics.csv`
- `speedup_vs_replication.png`
- `elapsed_ms_vs_replication_linear_parallel.png`
- `throughput_vs_replication.png`

## Notes

- Speedup is computed as mean linear elapsed divided by mean parallel elapsed for each (agent, replication_factor).
- Ranking score is mean speedup across replication factors greater than 1.