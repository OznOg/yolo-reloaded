#include "gemm.hpp"

void gemm_nn(size_t M, size_t N, size_t K, float ALPHA, 
             const float *__restrict__ A, int lda, 
             const float *__restrict__ B, int ldb,
             float *__restrict__ C, int ldc) {
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float A_PART = ALPHA * A[i * lda + k];
            for (size_t j = 0; j < N; ++j) {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

void gemm_nt(size_t M, size_t N, size_t K, float ALPHA, 
             const float *__restrict__ A, int lda, 
             const float *__restrict__ B, int ldb,
             float *__restrict__ C, int ldc) {
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0;
            for (size_t k = 0; k < K; ++k) {
                sum += ALPHA * A[i * lda + k] * B[j * ldb + k];
            }
            C[i * ldc + j] += sum;
        }
    }
}

void gemm_tn(size_t M, size_t N, size_t K, float ALPHA, 
             const float *__restrict__ A, int lda, 
             const float *__restrict__ B, int ldb,
             float *__restrict__ C, int ldc) {
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float A_PART = ALPHA * A[k * lda + i];
            for (size_t j = 0; j < N; ++j) {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

void gemm_tt(size_t M, size_t N, size_t K, float ALPHA, 
             const float *__restrict__ A, int lda, 
             const float *__restrict__ B, int ldb,
             float *__restrict__ C, int ldc) {
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0;
            for (size_t k = 0; k < K; ++k) {
                sum += ALPHA * A[i + k * lda] * B[k + j * ldb];
            }
            C[i * ldc + j] += sum;
        }
    }
}


