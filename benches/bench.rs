use divan::black_box;
use simd_test::nanmean_pulp;

fn main() {
    // Run registered benchmarks.
    divan::main();
}

mod nanmean {
    use super::*;
    use aligned_vec::{avec, AVec};
    use divan::counter::BytesCount;
    use simd_test::nanmean_c_avx512;

    const N: usize = 131072;

    #[divan::bench]
    fn pulp(bencher: divan::Bencher) {
        bencher
            .counter(BytesCount::new(N * 4))
            .with_inputs(|| -> AVec<f32> { avec![0f32; N] })
            .input_counter(|x: &AVec<f32>| BytesCount::of_slice(x))
            .bench_refs(|x| nanmean_pulp(black_box(x)))
    }

    #[divan::bench]
    fn c_avx512(bencher: divan::Bencher) {
        bencher
            .counter(BytesCount::new(N * 4))
            .with_inputs(|| -> AVec<f32> { avec![0f32; N] })
            .input_counter(|x: &AVec<f32>| BytesCount::of_slice(x))
            .bench_refs(|x| nanmean_c_avx512(black_box(x)))
    }
}
