#pragma once
#include <cstddef>

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

template <bool transpose_A, bool transpose_B>
void gemm(size_t M, size_t N, size_t K, float ALPHA,
          const float *__restrict__ A, int lda, 
          const float *__restrict__ B, int ldb, float BETA,
          float *__restrict__ C, int ldc)
{
    gemm_cpu<transpose_A, transpose_B>(M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}

