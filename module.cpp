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
                           const int sizeX, const vec_t& val) {
  size_t offset = x * sizeX + y;
  assert(offset < tensor.size());
  for (int i = 0; i < 8; ++ i) {
    twoDimWrite(tensor, x, y +i, sizeX, val.d[i]);
    assert(twoDimRead(tensor, x, y +i, sizeX) >= 0.00001);
  }
  // _mm256_storeu_ps((tensor.data() + offset), val);
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

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor) {
  tensor = tensor.flatten();
  tensor = tensor.contiguous();
  std::vector<float> vec(tensor.data_ptr<float>(),
                         tensor.data_ptr<float>() + tensor.numel());
  return vec;
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
inline vec_t fourDimGetVecAlongD(const std::vector<float> &tensor, const int x,
                                 const int y, const int z, int b,
                                 const int sizeX, const int sizeY,
                                 const int sizeZ, int axis = 0) {
  // float vec[8];
  vec_t vec;
  vec.v =
      _mm256_loadu_ps(fourDimReadPtr(tensor, x, y, z, b, sizeX, sizeY, sizeZ));
  return vec;
}

inline vec_t fourDimGetVecAlongN(const std::vector<float> &tensor, const int x,
                                 const int y, const int z, int b,
                                 const int sizeX, const int sizeY,
                                 const int sizeZ, int axis = 0) {
  // float vec[8];
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

inline void printVec(const vec_t& vec) {
  for (int i = 0; i < 8; ++ i) {
    // if (vec.d[i] == 0) {
      std::cout << vec.d[i] << " ";
    // }
      // std::cout << vec.d[i] << " ";
  }
  std::cout << "\n";
}

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

  /* Here is an example of how to read/write 0's to  Q (B, H, N, d) using the 4D
     accessors

      //loop over Batch Size
       for (int b = 0; b < B; b++) {

           //loop over Heads
           for (int h = 0; h < H; h++) {

               //loop over Sequence Length
               for (int i = 0; i < N; i++) {

                   //loop over Embedding Dimensionality
                   for (int j = 0; j < d; j++) {
                      float val = fourDimRead(Q, b, h, i, j, H, N, d);
                      val = 0.0;
                      fourDimWrite(Q, b, h, i, j, H, N, d, val);
                   }
               }
           }
       }
  */

  /* Here is an example of how to read/write 0's to  QK_t (N, N) using the 2D
     accessors

         for (int i = 0; i < N; i++) {
             for (int j = 0; j < N; j++) {
                 float val = twoDimRead(QK_t, i, j, N);
             val = 0.0;
                 twoDimWrite(QK_t, i, j, N, val);
           }
       }
  */

  // -------- YOUR CODE HERE  -------- //
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      // Q(N, d) * K(N, d)
      for (int i = 0; i < N; ++i) {
        int last = (N / 8) * 8;
        // calculate:
        // qk(i, j), qk(i, j + 1), qk(i, j + 2), qk(i, j + 3)
        // qk(i, j + 4), qk(i, j + 5), qk(i, j + 6), qk(i, j + 7)
        for (int j = 0; j < last; j += 8) {
          vec_t qk_vec;
          qk_vec.v = _mm256_loadu_ps(twoDimReadPtr(QK_t, i, j, N));
          for (int k = 0; k < d; ++k) {
            float q_val = fourDimRead(Q, b, h, i, k, H, N, d);
            auto k_vec = fourDimGetVecAlongN(K, b, h, j, k, H, N, d);
            auto q_vec = _mm256_broadcast_ss(&q_val);
            qk_vec.v = _mm256_fmadd_ps(q_vec, k_vec.v, qk_vec.v);
          }
          twoDimWriteVec(QK_t, i, j, N, qk_vec);
        }
        // calculate qk(i, p)
        for (int p = last; p < N; ++p) {
          float sum = twoDimRead(QK_t, i, p, N);
          for (int k = 0; k < d; ++k) {
            float q_val = fourDimRead(Q, b, h, i, k, H, N, d);
            float k_val = fourDimRead(K, b, h, p, k, H, N, d);
            sum += q_val * k_val;
          }

          twoDimWrite(QK_t, i, p, N, sum);
        }
      }

      // Softmax(QK_t)
      for (int i = 0; i < N; ++i) {
        float exp_sum = 0.0f;
        int last = (N / 8) * 8;
        vec_t qkt_vec;
        for (int j = 0; j < last; j += 8) {
          qkt_vec.v = _mm256_loadu_ps(twoDimReadPtr(QK_t, i, j, N));
          exp_sum += (std::exp(qkt_vec.d[0]) + std::exp(qkt_vec.d[1]) +
                      std::exp(qkt_vec.d[2]) + std::exp(qkt_vec.d[3]) +
                      std::exp(qkt_vec.d[4]) + std::exp(qkt_vec.d[5]) +
                      std::exp(qkt_vec.d[6]) + std::exp(qkt_vec.d[7]));
        }
        for (int p = last; p < N; ++p) {
          float val = twoDimRead(QK_t, i, p, N);
          exp_sum += std::exp(val);
        }
        vec_t val_vec;
        vec_t softmax_vec;
        for (int j = 0; j < last; j += 8) {
          qkt_vec.v = _mm256_loadu_ps(twoDimReadPtr(QK_t, i, j, N));
          // val_vec.v = _mm256_loadu_ps(twoDimReadPtr(QK_t, i, j, N));
          softmax_vec.d[0] = std::exp(qkt_vec.d[0]) / exp_sum;
          softmax_vec.d[1] = std::exp(qkt_vec.d[1]) / exp_sum;
          softmax_vec.d[2] = std::exp(qkt_vec.d[2]) / exp_sum;
          softmax_vec.d[3] = std::exp(qkt_vec.d[3]) / exp_sum;
          softmax_vec.d[4] = std::exp(qkt_vec.d[4]) / exp_sum;
          softmax_vec.d[5] = std::exp(qkt_vec.d[5]) / exp_sum;
          softmax_vec.d[6] = std::exp(qkt_vec.d[6]) / exp_sum;
          softmax_vec.d[7] = std::exp(qkt_vec.d[7]) / exp_sum;
          twoDimWriteVec(QK_t, i, j, N, softmax_vec);
        }
        for (int p = last; p < N; ++p) {
          float val = twoDimRead(QK_t, i, p, N);
          float sotmax_val = std::exp(val) / exp_sum;
          twoDimWrite(QK_t, i, p, N, sotmax_val);
        }
      }

      // Softmax(QK_t)[N, N] * V[N, d]
      for (int i = 0; i < N; ++i) {
        int last = (d / 8) * 8;
        // o(i, j), o(i, j + 1), o(i, j + 2), o(i, j + 3)
        // o(i, j + 4), o(i, j + 5), o(i, j + 6), o(i, j + 7)
        for (int j = 0; j < last; j += 8) {
          vec_t output_vec;
          output_vec.v = _mm256_loadu_ps(twoDimReadPtr(O, i, j, d));
          for (int k = 0; k < N; ++k) {
            float qkt_val = twoDimRead(QK_t, i, k, N);
            auto v_vec = fourDimGetVecAlongD(V, b, h, k, j, H, N, d);
            auto qkt_vec = _mm256_broadcast_ss(&qkt_val);
            output_vec.v = _mm256_fmadd_ps(qkt_vec, v_vec.v, output_vec.v);
          }
          twoDimWriteVec(O, i, j, d, output_vec);
          // printVec(output_vec);
        }

        for (int p = last; p < d; ++p) {
          float sum = 0.0f;
          for (int k = 0; k < N; ++k) {
            std::cout << __func__ << "\n";
            float qkv_val = twoDimRead(QK_t, i, k, N);
            float v_val = fourDimRead(V, b, h, k, p, H, N, d);
            sum += qkv_val * v_val;
          }
          twoDimWrite(O, i, p, d, sum);
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
