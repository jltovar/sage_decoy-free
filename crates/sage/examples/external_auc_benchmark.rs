//! Bounded synthetic benchmark. No scientific files or fitting are accessed.
#[cfg(feature = "bench")]
fn main() {
    use sage_core::ml::external_auc::{
        benchmark_serial_auc, bounded_pairwise_reference, exact_external_auc,
    };
    use std::{hint::black_box, time::Instant};
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 5, "workers good-count null-count repetitions");
    let workers: usize = args[1].parse().unwrap();
    let good_n: usize = args[2].parse().unwrap();
    let null_n: usize = args[3].parse().unwrap();
    let repeats: usize = args[4].parse().unwrap();
    assert!(
        (1..=std::thread::available_parallelism().unwrap().get()).contains(&workers)
            && good_n <= 3_000_000
            && null_n <= 12_000_000
            && (1..=20).contains(&repeats)
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .unwrap();
    // Linux process CPU (all workers), distinct from wall time and whole-host %.
    // Optional on other platforms; this instrumentation is benchmark-only.
    let ticks = std::process::Command::new("getconf")
        .arg("CLK_TCK")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<f64>().ok());
    let cpu = || -> Option<f64> {
        let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
        let fields: Vec<_> = stat.rsplit_once(')')?.1.split_whitespace().collect();
        Some((fields.get(11)?.parse::<f64>().ok()? + fields.get(12)?.parse::<f64>().ok()?) / ticks?)
    };
    let mut state = 42_u64;
    let mut values = |n| {
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                // Repeated values, negative values and exact ties, deterministic order.
                ((state >> 32) % 100_003) as f64 / 1000.0 - 50.0
            })
            .collect::<Vec<_>>()
    };
    let good = values(good_n);
    let null = values(null_n);
    let small_g = &good[..good.len().min(2000)];
    let small_n = &null[..null.len().min(3000)];
    let start = Instant::now();
    let reference = bounded_pairwise_reference(small_g, small_n, true);
    let old_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let small_new = benchmark_serial_auc(small_g, small_n, true);
    let new_seconds = start.elapsed().as_secs_f64();
    assert_eq!(reference.to_bits(), small_new.to_bits());
    let measure = |parallel| {
        let cpu_before = cpu();
        let start = Instant::now();
        let mut outputs = Vec::new();
        for _ in 0..repeats {
            for higher in [true, true, true, false, false, false] {
                outputs.push(black_box(if parallel {
                    exact_external_auc(&good, &null, higher)
                } else {
                    benchmark_serial_auc(&good, &null, higher)
                }));
            }
        }
        (
            start.elapsed().as_secs_f64(),
            cpu().zip(cpu_before).map(|(a, b)| a - b),
            outputs,
        )
    };
    let (serial_seconds, serial_cpu_seconds, expected) = measure(false);
    let (parallel_seconds, parallel_cpu_seconds, actual) = pool.install(|| measure(true));
    assert_eq!(expected, actual);
    println!(
        "{}",
        serde_json::json!({"schema":2,"workload":"six synthetic external AUCs per repetition","workers":pool.install(rayon::current_num_threads),"good_n":good_n,"null_n":null_n,"repetitions":repeats,"bounded_pairwise_seconds":old_seconds,"bounded_new_serial_seconds":new_seconds,"bounded_pairs":small_g.len()*small_n.len(),"serial_seconds":serial_seconds,"parallel_seconds":parallel_seconds,"serial_cpu_seconds":serial_cpu_seconds,"parallel_cpu_seconds":parallel_cpu_seconds,"average_busy_logical_cpus":parallel_cpu_seconds.map(|t|t/parallel_seconds),"parallel_speedup":serial_seconds/parallel_seconds,"scientific_values_equal":true,"auc_higher":actual[0],"auc_lower":actual[3]})
    );
}

#[cfg(not(feature = "bench"))]
fn main() {
    eprintln!("Enable --features bench for the bounded synthetic benchmark");
}
