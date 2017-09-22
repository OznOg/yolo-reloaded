#pragma once

void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out);
void flatten(float *x, int size, int layers, int batch, int forward);
void weighted_sum_cpu(float *a, float *b, float *s, int n, float *c);
void weighted_delta_cpu(float *a, float *b, float *s, float *da, float *db, float *ds, int n, float *dc);
void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out);
void mean_cpu(float *x, int batch, int filters, int spatial, float *mean);
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
void const_cpu(int N, float ALPHA, float *X, int INCX);
void mul_cpu(int N, float *X, int INCX, float *Y, int INCY);
void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void scal_cpu(int N, float ALPHA, float *X, int INCX);
void fill_cpu(int N, float ALPHA, float *X, int INCX);
void deinter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT);
void inter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT);
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
void mult_add_into_cpu(int N, float *X, float *Y, float *Z);
void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error);
void l1_cpu(int n, float *pred, float *truth, float *delta, float *error);
void l2_cpu(int n, float *pred, float *truth, float *delta, float *error);
float dot_cpu(int N, float *X, int INCX, float *Y, int INCY);
void softmax(float *input, int n, float temp, int stride, float *output);
void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output);
