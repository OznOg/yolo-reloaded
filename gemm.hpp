#pragma once
#include <cstddef>
#include <cstring> //memset

void gemm_nn(size_t M, size_t N, size_t K, float ALPHA,
             const float *__restrict__ A, int lda,
             const float *__restrict__ B, int ldb,
             float *__restrict__ C, int ldc);

void gemm_nt(size_t M, size_t N, size_t K, float ALPHA,
             const float *__restrict__ A, int lda,
             const float *__restrict__ B, int ldb,
             float *__restrict__ C, int ldc);

void gemm_tn(size_t M, size_t N, size_t K, float ALPHA,
             const float *__restrict__ A, int lda,
             const float *__restrict__ B, int ldb,
             float *__restrict__ C, int ldc);

void gemm_tt(size_t M, size_t N, size_t K, float ALPHA,
             const float *__restrict__ A, int lda,
             const float *__restrict__ B, int ldb,
             float *__restrict__ C, int ldc);

template <bool transpose_A, bool transpose_B>
static inline void gemm_cpu(size_t M, size_t N, size_t K, float ALPHA,
                            const float *__restrict__ A, int lda,
			    const float *__restrict__ B, int ldb, float BETA,
			    float *__restrict__ C, int ldc)
{
    if (BETA != 1)
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                C[i*ldc + j] *= BETA;
            }
        }
    if (not transpose_A and not transpose_B)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if (transpose_A and not transpose_B)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(not transpose_A and transpose_B)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

/*    <---- K ---->  <---- N ---->           <---- N ---->
 * ^  /          \    /         \   ^     ^   /          \
 * | |            |  |           |  |     |  |            |
 * | |            |  |  Matrix B |  K     |  |            |
 * M |   Matrix A |  |           |  |     M  |  Matrix C  |
 * | |            |   \         /   V     |  |            |
 * | |            |                       |  |            |
 * V  \          /                        V   \          /
 *
 *    Result is
 *               C = α A * B + β C
 */

template <bool transpose_A, bool transpose_B>
void gemm(size_t M, size_t N, size_t K, float ALPHA,
          const float *__restrict__ A, int lda,
          const float *__restrict__ B, int ldb, float BETA,
          float *__restrict__ C, int ldc)
{
    gemm_cpu<transpose_A, transpose_B>(M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}

#include <vector>

static inline void im2col_cpu(const std::vector<float> &data_im,
        int channels,  int height,  int width,
        int ksize,  int stride, int pad, float *data_col)
{
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col =  (width  + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
#pragma omp parallel for firstprivate(channels_col, width_col, height_col)
    for (int c = 0; c < channels_col; ++c) {

        int w_offset = c % ksize;

        int h_offset = (c / ksize) % ksize;

        int c_im = c / ksize / ksize;

        for (int h = 0; h < height_col; ++h) {
            int row = h_offset + h * stride - pad;
            int col_index = (c * height_col + h) * width_col;
            if (row < 0 || row >= height) {
                std::memset(&data_col[col_index], 0, width_col);
            } else
                for (int w = 0; w < width_col; ++w) {
                    int col = w_offset + w * stride - pad;

                    if (col < 0 || col >= width)
                        data_col[col_index + w] = 0;
                    else
                        data_col[col_index + w] = data_im[col + width * (row + height * c_im)];
                }
        }
    }
}

