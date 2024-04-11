use libc::size_t;
use pulp::Simd;

#[pulp::with_simd(nanmean_pulp = pulp::Arch::new())]
#[inline(always)]
pub fn nanmean_with_simd<S: Simd>(simd: S, v: &[f32]) -> f32 {
    // Constants
    let zeros = simd.f32s_splat(0.);
    let ones = simd.f32s_splat(1.);

    // Four accumulators to leverage instruction-level parallelism
    let mut count0 = simd.f32s_splat(0.0);
    let mut count1 = simd.f32s_splat(0.0);
    let mut count2 = simd.f32s_splat(0.0);
    let mut count3 = simd.f32s_splat(0.0);

    let mut sum0 = simd.f32s_splat(0.0);
    let mut sum1 = simd.f32s_splat(0.0);
    let mut sum2 = simd.f32s_splat(0.0);
    let mut sum3 = simd.f32s_splat(0.0);

    // Split input chunks
    let (head, tail) = S::f32s_as_simd(v);
    let (head4, head1) = pulp::as_arrays::<4, _>(head);

    // Deal with the 4 simd chunk, chunks
    for &[x0, x1, x2, x3] in head4 {
        // Create the NaN masks
        let mask0 = simd.f32s_equal(x0, x0);
        let mask1 = simd.f32s_equal(x1, x1);
        let mask2 = simd.f32s_equal(x2, x2);
        let mask3 = simd.f32s_equal(x3, x3);

        // Accumulate counts
        count0 = simd.f32s_add(simd.m32s_select_f32s(mask0, ones, zeros), count0);
        count1 = simd.f32s_add(simd.m32s_select_f32s(mask1, ones, zeros), count1);
        count2 = simd.f32s_add(simd.m32s_select_f32s(mask2, ones, zeros), count2);
        count3 = simd.f32s_add(simd.m32s_select_f32s(mask3, ones, zeros), count3);

        // Accumulate masked values
        sum0 = simd.f32s_add(simd.m32s_select_f32s(mask0, x0, zeros), sum0);
        sum1 = simd.f32s_add(simd.m32s_select_f32s(mask1, x1, zeros), sum1);
        sum2 = simd.f32s_add(simd.m32s_select_f32s(mask2, x2, zeros), sum2);
        sum3 = simd.f32s_add(simd.m32s_select_f32s(mask3, x3, zeros), sum3);
    }

    // Then deal with the rest of the chunk
    for &x0 in head1 {
        let mask0 = simd.f32s_equal(x0, x0);
        let masked_ones0 = simd.m32s_select_f32s(mask0, ones, zeros);
        count0 = simd.f32s_add(masked_ones0, count0);
        let masked_chunk0 = simd.m32s_select_f32s(mask0, x0, zeros);
        sum0 = simd.f32s_add(masked_chunk0, sum0);
    }

    // Parallel reduce the sums and counts
    let sum0 = simd.f32s_add(sum0, sum1);
    let sum2 = simd.f32s_add(sum2, sum3);
    let sum0 = simd.f32s_add(sum0, sum2);
    let mut sum = simd.f32s_reduce_sum(sum0);

    let count0 = simd.f32s_add(count0, count1);
    let count2 = simd.f32s_add(count2, count3);
    let count0 = simd.f32s_add(count0, count2);
    let mut count = simd.f32s_reduce_sum(count0);

    tail.iter().for_each(|x| {
        if !x.is_nan() {
            count += 1.0;
            sum += x;
        }
    });

    // Return the mean
    sum / count
}

#[link(name = "nanmean")]
extern "C" {
    fn nanmean(nums: *const f32, n: size_t) -> f32;
}

pub fn nanmean_c_avx512(v: &[f32]) -> f32 {
    unsafe { nanmean(v.as_ptr(), v.len() as size_t) }
}
