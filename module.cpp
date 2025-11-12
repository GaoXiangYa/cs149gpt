#include <ATen/ATen.h>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <sys/time.h>
#include <time.h>
#include <torch/extension.h>
#include <vector>
#include <xmmintrin.h>
#include <x86intrin.h>
// Uncomment for ISPC
// #include "module_ispc.h"
// using namespace ispc;

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

typedef union {
  float d[8];
  __m256 v;
} vec_t;

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(const std::vector<float> &tensor, const int x,
                        const int y, const int &sizeX) {
  // Note that sizeX is the size of a Row, not the number of rows
  return tensor[x * (sizeX) + y];
}

inline void twoDimWrite(std::vector<float> &tensor, const int x, const int y,
                        const int sizeX, const float val) {
  tensor[x * (sizeX) + y] = val;
}

inline int twoDimOffset(const std::vector<float> &tensor, const int x,
                        const int y, const int &sizeX) {
  auto offset = x * sizeX + y;
  assert(offset < tensor.size());
  return offset;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(const std::vector<float> &tensor, const int x,
                         const int y, const int z, int b, const int sizeX,
                         const int sizeY, const int sizeZ) {
  return tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * sizeZ +
                b];
}

inline void fourDimWrite(std::vector<float> &tensor, const int x, const int y,
                         const int z, const int b, const int sizeX,
                         const int sizeY, const int sizeZ, const float val) {
  tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * sizeZ + b] =
      val;
}

inline int fourDimOffset(const std::vector<float> &tensor, const int x,
                         const int y, const int z, int b, const int sizeX,
                         const int sizeY, const int sizeZ) {
  size_t offset =
      x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * sizeZ + b;
  assert(offset < tensor.size());
  return offset;
}

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor) {
  tensor = tensor.flatten();
  tensor = tensor.contiguous();
  std::vector<float> vec(tensor.data_ptr<float>(),
                         tensor.data_ptr<float>() + tensor.numel());
  return vec;
}

inline __m256 fastVecExp(const vec_t &x) {
  const __m256 r = _mm256_set1_ps(0x1.8p23f);
  const __m256 c_log2e = _mm256_set1_ps(0x1.715476p+0f); // log2(e)
  const __m256 c_magic = _mm256_set1_ps(0x1.8p23f);
  const __m256 c_invln2_approx = _mm256_set1_ps(0x1.7f7d1cp-20f);
  const __m256 c_ln2_hi = _mm256_set1_ps(0x1.62e4p-1f);
  const __m256 c_192 = _mm256_set1_ps(192.0f);
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 zero = _mm256_setzero_ps();

  // z = x * log2(e) + r
  const __m256 z = _mm256_fmadd_ps(x.v, c_log2e, r);
  const __m256 n = _mm256_sub_ps(z, r);

  // b = -(n * c_invln2_approx) - (n * c_ln2_hi - x)
  const __m256 t1 = _mm256_fnmadd_ps(n, c_invln2_approx, zero);
  const __m256 b = _mm256_fnmadd_ps(n, c_ln2_hi, x.v);

  // |n| > 192 → mask for overflow
  const __m256 abs_n = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), n);
  const __m256 mask_over = _mm256_cmp_ps(abs_n, c_192, _CMP_GT_OQ);

  // u = b * b
  const __m256 u = _mm256_mul_ps(b, b);

  // 多项式逼近 e^b ≈ 1 + b + b² * (...)，来自 ggml 的系数
  const __m256 c1 = _mm256_set1_ps(0x1.0e4020p-7f);
  const __m256 c2 = _mm256_set1_ps(0x1.573e2ep-5f);
  const __m256 c3 = _mm256_set1_ps(0x1.555e66p-3f);
  const __m256 c4 = _mm256_set1_ps(0x1.fffdb6p-2f);
  const __m256 c5 = _mm256_set1_ps(0x1.ffffecp-1f);

  __m256 j = _mm256_fmadd_ps(c1, b, c2);
  j = _mm256_fmadd_ps(j, u, _mm256_fmadd_ps(c3, b, c4));
  j = _mm256_fmadd_ps(j, u, _mm256_fmadd_ps(c5, b, one));

  // n 的整数部分表示 2^n
  // AVX2 没有 _mm256_scalef_ps，用手动实现
  __m256 n_int = _mm256_floor_ps(n);
  __m256 pow2n = _mm256_castsi256_ps(_mm256_slli_epi32(
      _mm256_add_epi32(_mm256_cvttps_epi32(n_int), _mm256_set1_epi32(127)),
      23));

  __m256 res = _mm256_mul_ps(j, pow2n);

  // 溢出处理
  __m256 alt = _mm256_blendv_ps(_mm256_set1_ps(INFINITY), zero,
                                _mm256_cmp_ps(n, zero, _CMP_LE_OQ));
  res = _mm256_blendv_ps(res, alt, mask_over);

  return res;
}

inline float getVecExpSum(const vec_t &vec) {
  vec_t res;
  res.v = fastVecExp(vec);
  float sum = 0.0f;
  for (int i = 0; i < 8; ++i) {
    sum += res.d[i];
  }
  return sum;
}

inline void getVecSoftmax(vec_t &softmax_vec, const vec_t &qkt_vec,
                          const float exp_sum) {
  softmax_vec.d[0] = std::exp(qkt_vec.d[0]) / exp_sum;
  softmax_vec.d[1] = std::exp(qkt_vec.d[1]) / exp_sum;
  softmax_vec.d[2] = std::exp(qkt_vec.d[2]) / exp_sum;
  softmax_vec.d[3] = std::exp(qkt_vec.d[3]) / exp_sum;
  softmax_vec.d[4] = std::exp(qkt_vec.d[4]) / exp_sum;
  softmax_vec.d[5] = std::exp(qkt_vec.d[5]) / exp_sum;
  softmax_vec.d[6] = std::exp(qkt_vec.d[6]) / exp_sum;
  softmax_vec.d[7] = std::exp(qkt_vec.d[7]) / exp_sum;
}

#define A(i, j) A[(i) * lda + j]
#define B(i, j) B[(i) * ldb + j]
#define C(i, j) C[(i) * ldc + j]

#define A_idx(i, j) (i) * lda + j

template <bool TransposeB = false>
inline void addDot4x8(float *A, float *B, float *C, int M, int N, int K,
                      const int lda, const int ldb, const int ldc) {
  constexpr float zero = 0.0f;
  vec_t a_vec0, a_vec1, a_vec2, a_vec3;
  vec_t b_vec;
  vec_t c_vec0, c_vec1, c_vec2, c_vec3;

  float a_val0 = 0.0f, a_val1 = 0.0f, a_val2 = 0.0f, a_val3 = 0.0f;

  c_vec0.v = _mm256_loadu_ps(&C(0, 0));
  c_vec1.v = _mm256_loadu_ps(&C(1, 0));
  c_vec2.v = _mm256_loadu_ps(&C(2, 0));
  c_vec3.v = _mm256_loadu_ps(&C(3, 0));

  a_vec0.v = _mm256_broadcast_ss(&zero);
  a_vec1.v = _mm256_broadcast_ss(&zero);
  a_vec2.v = _mm256_broadcast_ss(&zero);
  a_vec3.v = _mm256_broadcast_ss(&zero);

  b_vec.v = _mm256_broadcast_ss(&zero);

  for (int k = 0; k < K; ++k) {
    a_val0 = A(0, k);
    a_val1 = A(1, k);
    a_val2 = A(2, k);
    a_val3 = A(3, k);
    a_vec0.v = _mm256_broadcast_ss(&a_val0);
    a_vec1.v = _mm256_broadcast_ss(&a_val1);
    a_vec2.v = _mm256_broadcast_ss(&a_val2);
    a_vec3.v = _mm256_broadcast_ss(&a_val3);

    b_vec.v = _mm256_load_ps(B + k * ldb);

    c_vec0.v = _mm256_fmadd_ps(a_vec0.v, b_vec.v, c_vec0.v);
    c_vec1.v = _mm256_fmadd_ps(a_vec1.v, b_vec.v, c_vec1.v);
    c_vec2.v = _mm256_fmadd_ps(a_vec2.v, b_vec.v, c_vec2.v);
    c_vec3.v = _mm256_fmadd_ps(a_vec3.v, b_vec.v, c_vec3.v);
  }

  _mm256_storeu_ps(&C(0, 0), c_vec0.v);
  _mm256_storeu_ps(&C(1, 0), c_vec1.v);
  _mm256_storeu_ps(&C(2, 0), c_vec2.v);
  _mm256_storeu_ps(&C(3, 0), c_vec3.v);
}

template <bool TransposeB = false>
inline void addDot4x1(float *A, float *B, float *C, const int M, const int N,
                      const int K, const int lda, const int ldb,
                      const int ldc) {
  float a_val0 = 0.0f, a_val1 = 0.0f, a_val2 = 0.0f, a_val3 = 0.0f;
  float b_val = 0.0f;
  float c_val0 = 0.0f, c_val1 = 0.0f, c_val2 = 0.0f, c_val3 = 0.0f;
  for (int k = 0; k < K; ++k) {
    a_val0 = A(0, k);
    a_val1 = A(1, k);
    a_val2 = A(2, k);
    a_val3 = A(3, k);

    b_val = !TransposeB ? B(k, 0) : B(0, k);

    c_val0 += a_val0 * b_val;
    c_val1 += a_val1 * b_val;
    c_val2 += a_val2 * b_val;
    c_val3 += a_val3 * b_val;
  }
  C(0, 0) = c_val0;
  C(1, 0) = c_val1;
  C(2, 0) = c_val2;
  C(3, 0) = c_val3;
}

template <bool IsColMajor>
inline void packMatrixOutter(float *A, float *packed_A, int m, int n, int lda) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      if constexpr (IsColMajor) {
        packed_A[j * n + i] = A(i, j);
      } else {
        *packed_A++ = A(i, j);
      }
    }
  }
}

// pack B to [m, n]
template <bool TransposeB>
inline void packMatrixInner(float *B, float *packedB, int m, int n, int ldb) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      *packedB++ = TransposeB ? B(j, i) : B(i, j);
    }
  }
}

void printMatrix(float *A, int m, int n, int lda) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      std::cout << A(i, j) << " ";
    }
    std::cout << "\n";
  }
}
// A[M, K] * B[K, N] = C[M, N]
template <bool TransposeB>
inline void innerMatMulKernel(float *A, float *B, float *C, const int M,
                              const int N, const int K, const int lda,
                              const int ldb, const int ldc) {
  constexpr float zero = 0.0f;
  constexpr int MR = 4;
  constexpr int NR = 8;

  int last = 0;

  alignas(32) float packedB[NR * K];

  for (int j = 0; j < N; j += NR) {
    int jb = std::min(N - j, NR);
    float *B_ptr = TransposeB ? &B(j, 0) : &B(0, j);
    packMatrixInner<TransposeB>(B_ptr, packedB, K, jb, ldb);
    for (int i = 0; i < M; i += MR) {
      addDot4x8(&A(i, 0), packedB, &C(i, j), M, N, K, lda, jb, ldc);
    }
  }
}

inline void innerSoftmaxKernel(float *A, const int M, const int N) {
  const int lda = N;
  constexpr int MR = 4;
  constexpr int NR = 8;
  constexpr float zero = 0.0f;

  float exp_sum0 = 0.0f, exp_sum1 = 0.0f, exp_sum2 = 0.0f, exp_sum3 = 0.0f;
  float qkt_val0 = 0.0f, qkt_val1 = 0.0f, qkt_val2 = 0.0f, qkt_val3 = 0.0f;

  int last = 0;

  vec_t qkt_vec0, qkt_vec1, qkt_vec2, qkt_vec3;
  vec_t softmax_vec0, softmax_vec1, softmax_vec2, softmax_vec3;
  // Softmax(QK_t)
  for (int i = 0; i < M; i += MR) {
    exp_sum0 = exp_sum1 = exp_sum2 = exp_sum3 = 0.0f;
    last = (N / NR) * NR;
    for (int j = 0; j < last; j += NR) {
      qkt_vec0.v = _mm256_loadu_ps(&A(i, j));
      qkt_vec1.v = _mm256_loadu_ps(&A(i + 1, j));
      qkt_vec2.v = _mm256_loadu_ps(&A(i + 2, j));
      qkt_vec3.v = _mm256_loadu_ps(&A(i + 3, j));

      exp_sum0 += getVecExpSum(qkt_vec0);
      exp_sum1 += getVecExpSum(qkt_vec1);
      exp_sum2 += getVecExpSum(qkt_vec2);
      exp_sum3 += getVecExpSum(qkt_vec3);
    }

    for (int p = last; p < N; ++p) {
      qkt_val0 = A(i, p);
      qkt_val1 = A(i + 1, p);
      qkt_val2 = A(i + 2, p);
      qkt_val3 = A(i + 3, p);

      exp_sum0 += std::exp(qkt_val0);
      exp_sum1 += std::exp(qkt_val1);
      exp_sum2 += std::exp(qkt_val2);
      exp_sum3 += std::exp(qkt_val3);
    }

    for (int j = 0; j < last; j += NR) {
      qkt_vec0.v = _mm256_loadu_ps(&A(i, j));
      qkt_vec1.v = _mm256_loadu_ps(&A(i + 1, j));
      qkt_vec2.v = _mm256_loadu_ps(&A(i + 2, j));
      qkt_vec3.v = _mm256_loadu_ps(&A(i + 3, j));

      getVecSoftmax(softmax_vec0, qkt_vec0, exp_sum0);
      getVecSoftmax(softmax_vec1, qkt_vec1, exp_sum1);
      getVecSoftmax(softmax_vec2, qkt_vec2, exp_sum2);
      getVecSoftmax(softmax_vec3, qkt_vec3, exp_sum3);

      _mm256_storeu_ps(&A(i, j), softmax_vec0.v);
      _mm256_storeu_ps(&A(i + 1, j), softmax_vec1.v);
      _mm256_storeu_ps(&A(i + 2, j), softmax_vec2.v);
      _mm256_storeu_ps(&A(i + 3, j), softmax_vec3.v);
    }

    for (int p = last; p < N; ++p) {
      qkt_val0 = A(i, p);
      qkt_val1 = A(i + 1, p);
      qkt_val2 = A(i + 2, p);
      qkt_val3 = A(i + 3, p);

      A(i, p) = std::exp(qkt_val0) / exp_sum0;
      A(i + 1, p) = std::exp(qkt_val1) / exp_sum1;
      A(i + 2, p) = std::exp(qkt_val2) / exp_sum2;
      A(i + 3, p) = std::exp(qkt_val3) / exp_sum3;
    }
  }
}

inline void innerBlockedSoftmaxKernel(std::vector<float> &QK_t, const int M,
                                      const int N) {
  // Softmax(QK_t)
  const int MC = 512;
  alignas(32) float packed_QKt[MC * N];

  for (int i = 0; i < M; i += MC) {
    int ib = std::min(M - i, MC);
    int jb = N;
    int QKt_offset = twoDimOffset(QK_t, i, 0, N);
    innerSoftmaxKernel(QK_t.data() + QKt_offset, ib, jb);
  }
}
/* Programming Your Attention Modules.
 *
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We
 * have also created O and QK^t Tensors that are formatted as vectors. After you
 * have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it
 * this way - if I asked my dnn a question and it output 5 different answers it
 * had a batch size of 5. These samples are independent of each other and thus
 * can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This
 * effectively allows each head to operate the same attention algorithm, but
 * each with each head using different hyperparameters. These allow each head to
 * have their own definition of what relevance is when looking at a token. These
 * heads can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the
 * number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per
 * attention head. Let's say I encoded a word using the follow (length, number
 * of vowels, has a capital letters). The emvedded dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor,
                               torch::Tensor VTensor, torch::Tensor QK_tTensor,
                               int B, int H, int N, int d) {

  // Q, K, V are passed in with Shape: (B, H, N, d)
  // QK^t Intermediate Tensor has Shape (N, N)

  // Make O Tensor with Shape (B, H, N, d)
  at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

  // Format O, Q, K, and V tensors into 4D vectors
  std::vector<float> O = formatTensor(OTensor);
  std::vector<float> Q = formatTensor(QTensor);
  std::vector<float> K = formatTensor(KTensor);
  std::vector<float> V = formatTensor(VTensor);

  // Format QK_t Tensor into a 2D vector.
  std::vector<float> QK_t = formatTensor(QK_tTensor);

  // -------- YOUR CODE HERE  -------- //
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      int Q_offset = fourDimOffset(Q, b, h, 0, 0, H, N, d);
      int K_offset = fourDimOffset(K, b, h, 0, 0, H, N, d);
      int QKt_offset = twoDimOffset(QK_t, 0, 0, N);
      // Q(N, d) * K(N, d)
      innerMatMulKernel<true>(Q.data() + Q_offset, K.data() + K_offset,
                              QK_t.data() + QKt_offset, N, N, d, d, d, N);

      // Softmax(QK_t)[N, N] * V[N, d]
      innerSoftmaxKernel(QK_t.data() + QKt_offset, N, N);

      int V_offset = fourDimOffset(V, b, h, 0, 0, H, N, d);
      int O_offset = fourDimOffset(O, b, h, 0, 0, H, N, d);
      innerMatMulKernel<false>(QK_t.data() + QKt_offset, V.data() + V_offset,
                               O.data() + O_offset, N, d, N, N, d, d);
    }
  }
  // DO NOT EDIT THIS RETURN STATEMENT //
  // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and
  // returns it //
  return torch::from_blob(O.data(), {B, H, N, d},
                          torch::TensorOptions().dtype(torch::kFloat32))
      .clone();
}

// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor,
                                        torch::Tensor KTensor,
                                        torch::Tensor VTensor,
                                        torch::Tensor QK_tTensor, int B, int H,
                                        int N, int d) {

  // Q, K, V are passed in with Shape: (B, H, N, d)
  // QK^t Intermediate Tensor has Shape (N, N)

  // Make O Tensor with Shape (B, H, N, d)
  at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

  // Format O, Q, K, and V tensors into 4D vectors
  std::vector<float> O = formatTensor(OTensor);
  std::vector<float> Q = formatTensor(QTensor);
  std::vector<float> K = formatTensor(KTensor);
  std::vector<float> V = formatTensor(VTensor);

  // Format QK_t Tensor into a 2D vector.
  std::vector<float> QK_t = formatTensor(QK_tTensor);

  // -------- YOUR CODE HERE  -------- //
  const int NC = 512;
  const int DC = 32;

  alignas(32) float packed_Q[NC * DC];
  alignas(32) float packed_K[NC * DC];
  alignas(32) float packed_QKt[NC * NC];
  alignas(32) float packed_V[NC * DC];

  // -------- YOUR CODE HERE  -------- //
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      
      for (int k = 0; k < d; k += DC) {
        int kb = std::min(d - k, DC);
        for (int i = 0; i < N; i += NC) {
          int ib = std::min(N - i, NC);
          // pack matrix Q[N, d]
          int Q_offset = fourDimOffset(Q, b, h, i, k, H, N, d);
          packMatrixOutter<false>(Q.data() + Q_offset, packed_Q, ib, kb, d);
          for (int j = 0; j < N; j += NC) {
            int jb = std::min(N - j, NC);
            // pack matrix K[N, d]
            int K_offset = fourDimOffset(K, b, h, j, k, H, N, d);

            int QKt_offset = twoDimOffset(QK_t, i, j, N);
            innerMatMulKernel<true>(packed_Q, K.data() + K_offset,
                                    QK_t.data() + QKt_offset, ib, jb, kb, kb, d,
                                    N);
          }
        }
      }

      // Softmax(QK_t)[N, N] * V[N, d]
      int QKt_offset = twoDimOffset(QK_t, 0, 0, N);
      innerBlockedSoftmaxKernel(QK_t, N, N);

      for (int k = 0; k < N; k += NC) {
        int kb = std::min(N - k, NC);
        for (int i = 0; i < N; i += NC) {
          int ib = std::min(N - i, NC);
          QKt_offset = twoDimOffset(QK_t, i, k, N);
          packMatrixOutter<false>(QK_t.data() + QKt_offset, packed_QKt, ib, kb,
                                  N);
          for (int j = 0; j < d; j += DC) {
            int jb = std::min(d - j, DC);
            // pack matrix V
            int V_offset = fourDimOffset(V, b, h, k, j, H, N, d);
            // packMatrixOutter<false>(V.data() + V_offset, packed_V, kb, jb,
            // d);

            // packed_QKt * packed_V
            int O_offset = fourDimOffset(O, b, h, i, j, H, N, d);

            __builtin_prefetch(packed_QKt, 1, 3);
            __builtin_prefetch(V.data() + V_offset, 1, 3);
            __builtin_prefetch(O.data() + O_offset, 0, 2);
            innerMatMulKernel<false>(packed_QKt, V.data() + V_offset,
                                     O.data() + O_offset, ib, jb, kb, kb, d,
                                     d);
          }
        }
      }
    }
  }

  // DO NOT EDIT THIS RETURN STATEMENT //
  // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and
  // returns it //
  return torch::from_blob(O.data(), {B, H, N, d},
                          torch::TensorOptions().dtype(torch::kFloat32))
      .clone();
}

// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor,
                               torch::Tensor VTensor, torch::Tensor temp, int B,
                               int H, int N, int d) {

  // Q, K, V are passed in with Shape: (B, H, N, d)

  // Make O Tensor with Shape (B, H, N, d)
  // and O Row Tensor with Shape (N)
  at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
  at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

  // Format Y, Q, K, and V tensors into 4D vectors
  std::vector<float> O = formatTensor(OTensor);
  std::vector<float> Q = formatTensor(QTensor);
  std::vector<float> K = formatTensor(KTensor);
  std::vector<float> V = formatTensor(VTensor);

  // Format ORow Tensor into a 1D vector
  //  You can simply access this as ORow[i]
  std::vector<float> ORow = formatTensor(ORowTensor);

  // -------- YOUR CODE HERE  -------- //
  // We give you a template of the first three loops for your convenience
  // loop over batch
  for (int b = 0; b < B; b++) {

    // loop over heads
    for (int h = 0; h < H; h++) {
      for (int i = 0; i < N; i++) {

        // YRow is moved inside so each OpenMP thread gets a local copy.
        at::Tensor ORowTensor = temp.index({torch::indexing::Slice(
            at::get_thread_num(), torch::indexing::None)});
        std::vector<float> ORow = formatTensor(ORowTensor);
        // YOUR CODE HERE
      }
    }
  }

  // DO NOT EDIT THIS RETURN STATEMENT //
  // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and
  // returns it //
  return torch::from_blob(O.data(), {B, H, N, d},
                          torch::TensorOptions().dtype(torch::kFloat32))
      .clone();
}

// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //

torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor,
                               torch::Tensor VTensor, torch::Tensor QiTensor,
                               torch::Tensor KjTensor, torch::Tensor VjTensor,
                               torch::Tensor SijTensor, torch::Tensor PijTensor,
                               torch::Tensor PVTensor, torch::Tensor OiTensor,
                               torch::Tensor LTensor, torch::Tensor LiTensor,
                               torch::Tensor LijTensor,
                               torch::Tensor LnewTensor, int Bc, int Br, int B,
                               int H, int N, int d) {

  // Q, K, V are passed in with Shape: (B, H, N, d)
  // Sij, Pij are passed in with Shape: (Br, Bc)
  // Kj, Vj are passed in with Shape: (Bc, d)
  // Qi, Oi, and PV  are passed in with Shape: (Br, d)
  // L in passed in with Shape: (N)
  // Li, Lij, and Lnew are passed in with shape (Br)

  // Make O Tensor with Shape (B, H, N, d)
  at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

  // Format All Tensors into Vectors
  std::vector<float> O = formatTensor(OTensor);
  std::vector<float> Q = formatTensor(QTensor);
  std::vector<float> K = formatTensor(KTensor);
  std::vector<float> V = formatTensor(VTensor);
  std::vector<float> Sij = formatTensor(SijTensor);
  std::vector<float> Pij = formatTensor(PijTensor);
  std::vector<float> Kj = formatTensor(KjTensor);
  std::vector<float> Vj = formatTensor(VjTensor);
  std::vector<float> Qi = formatTensor(QiTensor);
  std::vector<float> Oi = formatTensor(OiTensor);
  std::vector<float> l = formatTensor(LTensor);
  std::vector<float> PV = formatTensor(PVTensor);
  std::vector<float> li = formatTensor(LiTensor);
  std::vector<float> lij = formatTensor(LijTensor);
  std::vector<float> lnew = formatTensor(LnewTensor);

  // -------- YOUR CODE HERE  -------- //

  // DO NOT EDIT THIS RETURN STATEMENT //
  // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and
  // returns it //
  return torch::from_blob(O.data(), {B, H, N, d},
                          torch::TensorOptions().dtype(torch::kFloat32))
      .clone();
}

/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
  m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked,
        " Blocked Unfused Attention");
  m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
  m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
  m.def("twoDimRead", &twoDimRead, "twoDimRead");
  m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
