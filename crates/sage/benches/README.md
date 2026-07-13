# Decoy-free performance benchmarks

Run the focused Criterion suite with:

```bash
cargo bench -p sage-core --features bench --bench decoy_free_perf
```

The workloads cover:

- preparation of five model-specific null pools from 100,000 synthetic PSM features;
- LFQ time warping for a 64-by-4,096 matrix;
- LFQ convolution for 64 traces of length 4,096.

## Refactor comparison

Measured on 2026-07-12 with Rust 1.91.1 on Apple Silicon. Criterion used a
one-second warm-up, three-second measurement window, and 20 samples.

| Workload | Before | After | Result |
| --- | ---: | ---: | ---: |
| Null-pool preparation | 75.886 ms | 5.534 ms | 92.6% faster (13.7x) |
| LFQ time warping | 179.21 us | 194.16 us | No statistically significant change |
| LFQ convolution | 951.57 us | 960.60 us | Within Criterion's noise threshold |

The LFQ changes target allocation pressure rather than arithmetic throughput:
time-warping now allocates one scratch row per call instead of one per matrix
row, and convolution reuses its output and accumulation buffers. The benchmark
does not currently measure allocation counts, so its LFQ result should be read
as throughput-neutral rather than as evidence that allocation pressure is
unchanged.
