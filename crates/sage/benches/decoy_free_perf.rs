use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sage_core::decoy_free_fdr::NullPoolBenchmark;
use sage_core::lfq::LfqScratchBenchmark;

fn null_pool_benchmark(c: &mut Criterion) {
    let state = NullPoolBenchmark::new(20_000, 5);
    c.bench_function("decoy_free/null_pools/100k_features_5_models", |b| {
        b.iter(|| black_box(state.build_all()))
    });
}

fn lfq_scratch_benchmarks(c: &mut Criterion) {
    let mut time_warp = LfqScratchBenchmark::new(64, 4_096);
    c.bench_function("lfq/time_warp/64x4096", |b| {
        b.iter(|| black_box(time_warp.apply_time_warps()))
    });

    let mut convolution = LfqScratchBenchmark::new(64, 4_096);
    c.bench_function("lfq/convolution/64x4096", |b| {
        b.iter(|| black_box(convolution.convolve_all_rows()))
    });
}

criterion_group!(benches, null_pool_benchmark, lfq_scratch_benchmarks);
criterion_main!(benches);
