#include "nanmean.h"

float nanmean(float nums[], size_t n) {
  const __m512 ones = _mm512_set1_ps(1.0);
  const __m512 zeros = _mm512_setzero_ps();

  __m512 vsum0 = _mm512_setzero_ps();
  __m512 vsum1 = _mm512_setzero_ps();
  __m512 vsum2 = _mm512_setzero_ps();
  __m512 vsum3 = _mm512_setzero_ps();

  __m512 vcount0 = _mm512_setzero_ps();
  __m512 vcount1 = _mm512_setzero_ps();
  __m512 vcount2 = _mm512_setzero_ps();
  __m512 vcount3 = _mm512_setzero_ps();

  for (int i = 0; i < n; i += 64) {
    __m512 chunk0 = _mm512_loadu_ps((__m512 *)&nums[i + 0]);
    __m512 chunk1 = _mm512_loadu_ps((__m512 *)&nums[i + 16]);
    __m512 chunk2 = _mm512_loadu_ps((__m512 *)&nums[i + 32]);
    __m512 chunk3 = _mm512_loadu_ps((__m512 *)&nums[i + 48]);

    __mmask16 mask0 = _mm512_cmpord_ps_mask(chunk0, chunk0);
    __mmask16 mask1 = _mm512_cmpord_ps_mask(chunk1, chunk1);
    __mmask16 mask2 = _mm512_cmpord_ps_mask(chunk2, chunk2);
    __mmask16 mask3 = _mm512_cmpord_ps_mask(chunk3, chunk3);

    vsum0 = _mm512_mask_add_ps(zeros, mask0, vsum0, chunk0);
    vsum1 = _mm512_mask_add_ps(zeros, mask1, vsum1, chunk1);
    vsum2 = _mm512_mask_add_ps(zeros, mask2, vsum2, chunk2);
    vsum3 = _mm512_mask_add_ps(zeros, mask3, vsum3, chunk3);

    vcount0 = _mm512_mask_add_ps(zeros, mask0, vcount0, ones);
    vcount1 = _mm512_mask_add_ps(zeros, mask1, vcount1, ones);
    vcount2 = _mm512_mask_add_ps(zeros, mask2, vcount2, ones);
    vcount3 = _mm512_mask_add_ps(zeros, mask3, vcount3, ones);
  }

  vsum0 = _mm512_add_ps(vsum0, vsum1);
  vsum2 = _mm512_add_ps(vsum2, vsum3);
  vsum0 = _mm512_add_ps(vsum0, vsum2);
  float sum = _mm512_reduce_add_ps(vsum0);

  vcount0 = _mm512_add_ps(vcount0, vcount1);
  vcount2 = _mm512_add_ps(vcount2, vcount3);
  vcount0 = _mm512_add_ps(vcount0, vcount2);
  float count = _mm512_reduce_add_ps(vcount0);

  return sum / count;
}
