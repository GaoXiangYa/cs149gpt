#include <ATen/ATen.h>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <immintrin.h>
#include <iostream>
#include <sys/time.h>
#include <time.h>
#include <torch/extension.h>
#include <vector>
#include <xmmintrin.h>

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

inline const float *twoDimReadPtr(const std::vector<float> &tensor, const int x,
                                  const int y, const int &sizeX) {
  // Note that sizeX is the size of a Row, not the number of rows
  size_t offset = x * sizeX + y;
  assert(offset < tensor.size());
  return (tensor.data() + offset);
}

inline void twoDimWrite(std::vector<float> &tensor, const int x, const int y,
                        const int sizeX, const float val) {
  tensor[x * (sizeX) + y] = val;
}

inline void twoDimWriteVec(std::vector<float> &tensor, const int x, const int y,
                           const int sizeX, const vec_t &val) {
  size_t offset = x * sizeX + y;
  assert(offset < tensor.size());
  _mm256_storeu_ps((tensor.data() + offset), val.v);
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

inline const float *fourDimReadPtr(const std::vector<float> &tensor,
                                   const int x, const int y, const int z, int b,
                                   const int sizeX, const int sizeY,
                                   const int sizeZ) {
  size_t offset =
      x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * sizeZ + b;
  assert(offset < tensor.size());
  return (tensor.data() + offset);
}

inline void fourDimWrite(std::vector<float> &tensor, const int x, const int y,
                         const int z, const int b, const int sizeX,
                         const int sizeY, const int sizeZ, const float val) {
  tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * sizeZ + b] =
      val;
}

inline void fourDimWriteVec(std::vector<float> &tensor, const int x,
                            const int y, const int z, const int b,
                            const int sizeX, const int sizeY, const int sizeZ,
                            const vec_t &val) {
  size_t offset =
      x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * sizeZ + b;
  assert(offset < tensor.size());
  _mm256_storeu_ps(tensor.data() + offset, val.v);
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

inline vec_t fourDimReadVecAlongD(const std::vector<float> &tensor, const int x,
                                  const int y, const int z, int b,
                                  const int sizeX, const int sizeY,
                                  const int sizeZ, int axis = 0) {
  vec_t vec;
  vec.v =
      _mm256_loadu_ps(fourDimReadPtr(tensor, x, y, z, b, sizeX, sizeY, sizeZ));
  return vec;
}

inline vec_t fourDimReadVecAlongN(const std::vector<float> &tensor, const int x,
                                  const int y, const int z, int b,
                                  const int sizeX, const int sizeY,
                                  const int sizeZ, int axis = 0) {
  vec_t vec;
  vec.d[0] = fourDimRead(tensor, x, y, z, b, sizeX, sizeY, sizeZ);
  vec.d[1] = fourDimRead(tensor, x, y, z + 1, b, sizeX, sizeY, sizeZ);
  vec.d[2] = fourDimRead(tensor, x, y, z + 2, b, sizeX, sizeY, sizeZ);
  vec.d[3] = fourDimRead(tensor, x, y, z + 3, b, sizeX, sizeY, sizeZ);
  vec.d[4] = fourDimRead(tensor, x, y, z + 4, b, sizeX, sizeY, sizeZ);
  vec.d[5] = fourDimRead(tensor, x, y, z + 5, b, sizeX, sizeY, sizeZ);
  vec.d[6] = fourDimRead(tensor, x, y, z + 6, b, sizeX, sizeY, sizeZ);
  vec.d[7] = fourDimRead(tensor, x, y, z + 7, b, sizeX, sizeY, sizeZ);

  return vec;
}

inline float getVecExpSum(const vec_t &vec) {
  return (std::exp(vec.d[0]) + std::exp(vec.d[1]) + std::exp(vec.d[2]) +
          std::exp(vec.d[3]) + std::exp(vec.d[4]) + std::exp(vec.d[5]) +
          std::exp(vec.d[6]) + std::exp(vec.d[7]));
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

#define A(i, j) A[(i * lda) + j]
#define B(i, j) B[(i * ldb) + j]
#define C(i, j) C[(i * ldc) + j]

// M = N, N = d, k = N
// lda = d, ldb = d, ldc = N
inline void addDot4x8(float *A, float *B, float *C, int M, int N, int K) {
  constexpr float zero = 0.0f;
  vec_t a_vec0, a_vec1, a_vec2, a_vec3;
  vec_t b_vec;
  vec_t c_vec0, c_vec1, c_vec2, c_vec3;

  const int lda = N, ldb = N, ldc = K;

  float a_val0 = 0.0f, a_val1 = 0.0f, a_val2 = 0.0f, a_val3 = 0.0f;

  c_vec0.v = _mm256_broadcast_ss(&zero);
  c_vec1.v = _mm256_broadcast_ss(&zero);
  c_vec2.v = _mm256_broadcast_ss(&zero);
  c_vec3.v = _mm256_broadcast_ss(&zero);
  
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

    b_vec.v = _mm256_set_ps(B(7, k), B(6, k), B(5, k), B(4, k), B(3, k),
                            B(2, k), B(1, k), B(0, k));

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

inline void addDot4x1(float *A, float *B, float *C, const int M, const int N, const int K) {
  const int lda = M, ldb = N, ldc = K;
  float a_val0 = 0.0f, a_val1 = 0.0f, a_val2 = 0.0f, a_val3 = 0.0f;
  float b_val = 0.0f;
  float c_val0 = 0.0f, c_val1 = 0.0f, c_val2 = 0.0f, c_val3 = 0.0f;
  for (int k = 0; k < K; ++k) {
    a_val0 = A(0, k);
    a_val1 = A(1, k);
    a_val2 = A(2, k);
    a_val3 = A(3, k);

    b_val = B(0, k);

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

// A[M, K] * B[K, N] = C[M, N]
// M = n, N = d, K = N;
inline void innerKernel(float *A, float *B, float *C, const int M, const int N, const int K) {
  constexpr float zero = 0.0f;
  constexpr int MR = 4;
  constexpr int NR = 8;
  const int lda = N, ldb = N, ldc = K;

  int last = 0;

  for (int i = 0; i < M; i += MR) {
    last = (N / NR) * NR;
    for (int j = 0; j < last; j += NR) {
      addDot4x8(&A(i, 0), &B(0, j), &C(i, j), M, N, K);
    }
    for (int p = last; p < N; ++p) {
      addDot4x1(&A(i, 0), &B(0, p), &C(i, p), M, N, K);
    }
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
  vec_t k_vec;
  vec_t q_vec0, q_vec1, q_vec2, q_vec3;
  vec_t v_vec;
  vec_t qkt_vec0, qkt_vec1, qkt_vec2, qkt_vec3;
  vec_t softmax_vec0, softmax_vec1, softmax_vec2, softmax_vec3;
  vec_t o_vec0, o_vec1, o_vec2, o_vec3;

  float q_val0 = 0.0f, q_val1 = 0.0f, q_val2 = 0.0f, q_val3 = 0.0f;
  float qkt_val0 = 0.0f, qkt_val1 = 0.0f, qkt_val2 = 0.0f, qkt_val3 = 0.0f;
  float k_val = 0.0f, v_val = 0.0f, qk_val = 0.0f;
  float o_val0 = 0.0f, o_val1 = 0.0f, o_val2 = 0.0f, o_val3 = 0.0f;
  float exp_sum0 = 0.0f, exp_sum1 = 0.0f, exp_sum2 = 0.0f, exp_sum3 = 0.0f;
  float exp_sum = 0.0f;
  float softmax_val = 0.0f;
  const float zero = 0.0f;
  int last = 0;

  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      // Q(N, d) * K(N, d)
      int Q_offset = fourDimOffset(Q, b, h, 0, 0, H, N, d);
      int K_offset = fourDimOffset(K, b, h, 0, 0, H, N, d);
      int QKt_offset = twoDimOffset(QK_t, 0, 0, N);
      // std::cout << Q_offset << " " << K_offset << " " <<QKt_offset << "\n";
      innerKernel(Q.data() +Q_offset, K.data() + K_offset, QK_t.data() + QKt_offset, N, d, N);

      // Softmax(QK_t)
      for (int i = 0; i < N; i += 4) {
        exp_sum0 = exp_sum1 = exp_sum2 = exp_sum3 = 0.0f;
        last = (N / 8) * 8;
        for (int j = 0; j < last; j += 8) {
          qkt_vec0.v = _mm256_loadu_ps(twoDimReadPtr(QK_t, i, j, N));
          qkt_vec1.v = _mm256_loadu_ps(twoDimReadPtr(QK_t, i + 1, j, N));
          qkt_vec2.v = _mm256_loadu_ps(twoDimReadPtr(QK_t, i + 2, j, N));
          qkt_vec3.v = _mm256_loadu_ps(twoDimReadPtr(QK_t, i + 3, j, N));

          exp_sum0 += getVecExpSum(qkt_vec0);
          exp_sum1 += getVecExpSum(qkt_vec1);
          exp_sum2 += getVecExpSum(qkt_vec2);
          exp_sum3 += getVecExpSum(qkt_vec3);
        }
        for (int p = last; p < N; ++p) {
          qkt_val0 = twoDimRead(QK_t, i, p, N);
          qkt_val1 = twoDimRead(QK_t, i + 1, p, N);
          qkt_val2 = twoDimRead(QK_t, i + 2, p, N);
          qkt_val3 = twoDimRead(QK_t, i + 3, p, N);

          exp_sum += (std::exp(qkt_val0) + std::exp(qkt_val1) +
                      std::exp(qkt_val2) + std::exp(qkt_val3));
        }
        for (int j = 0; j < last; j += 8) {
          qkt_vec0.v = _mm256_loadu_ps(twoDimReadPtr(QK_t, i, j, N));
          qkt_vec1.v = _mm256_loadu_ps(twoDimReadPtr(QK_t, i + 1, j, N));
          qkt_vec2.v = _mm256_loadu_ps(twoDimReadPtr(QK_t, i + 2, j, N));
          qkt_vec3.v = _mm256_loadu_ps(twoDimReadPtr(QK_t, i + 3, j, N));

          getVecSoftmax(softmax_vec0, qkt_vec0, exp_sum0);
          twoDimWriteVec(QK_t, i, j, N, softmax_vec0);

          getVecSoftmax(softmax_vec1, qkt_vec1, exp_sum1);
          twoDimWriteVec(QK_t, i + 1, j, N, softmax_vec1);

          getVecSoftmax(softmax_vec2, qkt_vec2, exp_sum2);
          twoDimWriteVec(QK_t, i + 2, j, N, softmax_vec2);

          getVecSoftmax(softmax_vec3, qkt_vec3, exp_sum3);
          twoDimWriteVec(QK_t, i + 3, j, N, softmax_vec3);
        }
        for (int p = last; p < N; ++p) {
          qkt_val0 = twoDimRead(QK_t, i, p, N);
          qkt_val1 = twoDimRead(QK_t, i + 1, p, N);
          qkt_val2 = twoDimRead(QK_t, i + 2, p, N);
          qkt_val3 = twoDimRead(QK_t, i + 3, p, N);

          softmax_val = std::exp(qkt_val0) / exp_sum;
          twoDimWrite(QK_t, i, p, N, softmax_val);

          softmax_val = std::exp(qkt_val1) / exp_sum;
          twoDimWrite(QK_t, i + 1, p, N, softmax_val);

          softmax_val = std::exp(qkt_val2) / exp_sum;
          twoDimWrite(QK_t, i + 2, p, N, softmax_val);

          softmax_val = std::exp(qkt_val3) / exp_sum;
          twoDimWrite(QK_t, i + 3, p, N, softmax_val);
        }
      }

      // Softmax(QK_t)[N, N] * V[N, d]
      // int V_offset = fourDimOffset(V, b, h, 0, 0, H, N, d);
      // int O_offset = fourDimOffset(O, b, h, 0, 0, H, N, d);


      for (int i = 0; i < N; i += 4) {
        last = (d / 8) * 8;
        // o(i, j), ......... ,o(i, j + 7)
        // o(i + 1, j), ......... ,o(i + 1, j + 7)
        // o(i + 2, j), ......... ,o(i + 2, j + 7)
        // o(i + 3, j), ......... ,o(i + 3, j + 7)
        for (int j = 0; j < last; j += 8) {
          o_vec0.v = _mm256_broadcast_ss(&zero);
          o_vec1.v = _mm256_broadcast_ss(&zero);
          o_vec2.v = _mm256_broadcast_ss(&zero);
          o_vec3.v = _mm256_broadcast_ss(&zero);
          for (int k = 0; k < N; ++k) {
            qkt_val0 = twoDimRead(QK_t, i, k, N);
            qkt_val1 = twoDimRead(QK_t, i + 1, k, N);
            qkt_val2 = twoDimRead(QK_t, i + 2, k, N);
            qkt_val3 = twoDimRead(QK_t, i + 3, k, N);

            v_vec = fourDimReadVecAlongD(V, b, h, k, j, H, N, d);

            qkt_vec0.v = _mm256_broadcast_ss(&qkt_val0);
            qkt_vec1.v = _mm256_broadcast_ss(&qkt_val1);
            qkt_vec2.v = _mm256_broadcast_ss(&qkt_val2);
            qkt_vec3.v = _mm256_broadcast_ss(&qkt_val3);

            o_vec0.v = _mm256_fmadd_ps(qkt_vec0.v, v_vec.v, o_vec0.v);
            o_vec1.v = _mm256_fmadd_ps(qkt_vec1.v, v_vec.v, o_vec1.v);
            o_vec2.v = _mm256_fmadd_ps(qkt_vec2.v, v_vec.v, o_vec2.v);
            o_vec3.v = _mm256_fmadd_ps(qkt_vec3.v, v_vec.v, o_vec3.v);
          }
          fourDimWriteVec(O, b, h, i, j, H, N, d, o_vec0);
          fourDimWriteVec(O, b, h, i + 1, j, H, N, d, o_vec1);
          fourDimWriteVec(O, b, h, i + 2, j, H, N, d, o_vec2);
          fourDimWriteVec(O, b, h, i + 3, j, H, N, d, o_vec3);
        }

        for (int p = last; p < d; ++p) {
          o_val0 = o_val1 = o_val2 = o_val3 = 0.0f;
          for (int k = 0; k < N; ++k) {
            qkt_val0 = twoDimRead(QK_t, i, k, N);
            qkt_val1 = twoDimRead(QK_t, i + 1, k, N);
            qkt_val2 = twoDimRead(QK_t, i + 2, k, N);
            qkt_val3 = twoDimRead(QK_t, i + 3, k, N);

            v_val = fourDimRead(V, b, h, k, p, H, N, d);

            o_val0 += qkt_val0 * v_val;
            o_val1 += qkt_val1 * v_val;
            o_val2 += qkt_val2 * v_val;
            o_val3 += qkt_val3 * v_val;
          }
          fourDimWrite(O, b, h, i, p, H, N, d, o_val0);
          fourDimWrite(O, b, h, i + 1, p, H, N, d, o_val1);
          fourDimWrite(O, b, h, i + 2, p, H, N, d, o_val2);
          fourDimWrite(O, b, h, i + 3, p, H, N, d, o_val3);
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
  const int NC = 256;
  const int DC = 256;
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      // Block Q, K, QK_t
      for (int i = 0; i < N; i += NC) {
        int ib = std::min(NC, N - i);
        for (int j = 0; j < N; j += NC) {
          int jb = std::min(NC, N - j);
          for (int k = 0; k < d; k += DC) {
            int kb = std::min(DC, N - k);

            int Q_offset = fourDimOffset(Q, b, h, i, j, H, N, d);
            int K_offset = fourDimOffset(K, b, h, j, k, H, N, d);
            int QKt_offset = twoDimOffset(QK_t, i, j, N);

            innerKernel(Q.data() + Q_offset, K.data() + K_offset,
                        QK_t.data() + QKt_offset, ib, jb, kb);
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
