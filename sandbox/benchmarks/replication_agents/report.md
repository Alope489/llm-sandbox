# Replication Benefit Benchmark Report

## Scope

- agents: `extractor, kb, processor, simulation`
- replication_factors: `1, 2, 4, 6, 8`
- repetitions_per_point: `1`

## Ranking by replication benefit

- `1. processor` mean_speedup_factor_gt1=`1.916214` max_speedup_factor_gt1=`2.332807`
- `2. kb` mean_speedup_factor_gt1=`1.735555` max_speedup_factor_gt1=`2.711134`
- `3. extractor` mean_speedup_factor_gt1=`1.566124` max_speedup_factor_gt1=`1.695112`
- `4. simulation` mean_speedup_factor_gt1=`1.497021` max_speedup_factor_gt1=`1.81583`

## Artifacts

- `metrics.csv`
- `speedup_vs_replication.png`
- `elapsed_ms_vs_replication_linear_parallel.png`
- `throughput_vs_replication.png`

## Notes

- Speedup is computed as mean linear elapsed divided by mean parallel elapsed for each (agent, replication_factor).
- Ranking score is mean speedup across replication factors greater than 1.