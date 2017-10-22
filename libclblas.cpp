#include "common.h"

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>
#include <iostream>
#include <string>
#include <vector>

#define WPT 4
#define TS 16

static const std::string im2col_kernel(R"KERNEL(
__kernel void im2col(const int n, const __global float *data_im,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col,
        float __global *data_col) {
    int index = get_group_id(0) + get_local_id(0);
    for(; index < n; index += get_local_size(0)) {
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        __global float* data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        const __global float* data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;
      
                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                     data_im_ptr[i * width + j] : 0;

                data_col_ptr += height_col * width_col;
            }
        }
    }
})KERNEL");

static const std::string gemm_kernel(R"KERNEL(
#define WPT 4
#define TS 16
#define RTS ((TS) / (WPT))
__kernel void gemm(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)

    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    // Initialise the accumulation registers
    float acc[WPT];
    for (int w = 0; w < WPT; w++) {
        acc[w] = 0.0f;
    }

    // Synchronise before loading the next tile
    barrier(CLK_LOCAL_MEM_FENCE);

    // Loop over all tiles
    const int numTiles = ceil(K / (float)TS);
    for (int t = 0; t < numTiles; t++) {

        const int tiledRow = TS * t;
        const int tiledCol = TS * t;
        // Load one tile of A and B into local memory
        for (int w=0; w<WPT; w++) {
            Asub[col + w * RTS][row] = (tiledCol + col + w*RTS) >= K || globalRow >= M ? 0 : A[(tiledCol + col + w*RTS) * M + globalRow];
            Bsub[col + w * RTS][row] = (globalCol + w*RTS) >= N || tiledRow + row >= K ? 0 : B[(globalCol + w*RTS) * K + tiledRow + row];
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (int k = 0; k < TS; k++) {
            for (int w = 0; w < WPT; w++) {
                acc[w] += Asub[k][row] * Bsub[col + w * RTS][k];
            }
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    for (int w=0; w<WPT; w++) {
        if ((globalCol + w * RTS) >= N || globalRow >= M)
            return;
        C[(globalCol + w * RTS) * M + globalRow] = acc[w];
    }
}
)KERNEL");

class Setup {
    Setup() {
       // get all platforms (drivers)
       std::vector<cl::Platform> all_platforms;
       cl::Platform::get(&all_platforms);
       if (all_platforms.empty()) {
           throw "No platforms found. Check OpenCL installation!\n";
       }
       cl::Platform default_platform = all_platforms[0];

       //get default device of the default platform
       std::vector<cl::Device> all_devices;
       default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
       if (all_devices.empty()) {
           throw " No devices found. Check OpenCL installation!\n";
       }
       cl::Device default_device = all_devices[0];

       cl::Context context{default_device};

       //create queue to which we will push commands for the device.
       cl::CommandQueue queue(context, default_device);

       _ctx = context;
       _queue = queue;

       cl::Program::Sources sources;

       std::string kernel_code = gemm_kernel + im2col_kernel;

       sources.push_back({kernel_code.c_str(), kernel_code.length()});

       _program = cl::Program(_ctx, sources);
       if (_program.build({default_device}) != CL_SUCCESS) {
           std::string error;
           _program.getBuildInfo(default_device, CL_PROGRAM_BUILD_LOG, &error);
           std::cerr << error << std::endl;
           throw "Error building kernel\n";
       }
    }

    ~Setup() {
    }

public:
    auto &getCtx() {
	return _ctx;
    }

    auto &getQueue() {
	return _queue;
    }

    auto &getProgram() {
	return _program;
    }
    static Setup &getInstance() {
	static Setup setup;
	return setup;
    }
private:
    cl::CommandQueue _queue;
    cl::Context _ctx;
    cl::Program _program;
};


// Matrix-multiplication using the clBlas library. This function copies the input matrices to the
// GPU, runs SGEMM, and copies the output matrix back to the CPU.
void libclblas(float* A, int lda, float* B, int ldb, float* C, int ldc, float ALPHA, float BETA,
               int K, int M, int N) {
    (void)lda;
    (void)ldb;
    (void)ldc;
    (void)ALPHA;
    (void)BETA;

    cl_int err;

    // Configure clBlas
    auto &instance = Setup::getInstance();

    auto &ctx = instance.getCtx();
    auto &queue = instance.getQueue();

    // Prepare OpenCL memory objects
    cl::Buffer bufA(ctx, CL_MEM_READ_ONLY, M*K*sizeof(*A), NULL, &err);
    cl::Buffer bufB(ctx, CL_MEM_READ_ONLY, K*N*sizeof(*B), NULL, &err);
    cl::Buffer bufC(ctx, CL_MEM_READ_WRITE, M*N*sizeof(*C), NULL, &err);

    // Copy matrices to the GPU (also C to erase the results of the previous run)
    err = queue.enqueueWriteBuffer(bufA, CL_TRUE, 0, M*K*sizeof(*A), A, NULL, NULL);
    err = queue.enqueueWriteBuffer(bufB, CL_TRUE, 0, K*N*sizeof(*B), B, NULL, NULL);
    err = queue.enqueueWriteBuffer(bufC, CL_TRUE, 0, M*N*sizeof(*C), C, NULL, NULL);

    cl::Kernel gemmm = cl::Kernel(instance.getProgram(), "gemm");
    gemmm.setArg(1, M);
    gemmm.setArg(0, N);
    gemmm.setArg(2, K);
    gemmm.setArg(4, bufA);
    gemmm.setArg(3, bufB);
    gemmm.setArg(5, bufC);

    queue.enqueueNDRangeKernel(gemmm, cl::NullRange, cl::NDRange(N + TS - (N % TS), (M + TS - (M % TS)) / WPT), cl::NDRange(TS, TS / WPT));

    queue.finish();

    // Copy the output matrix C back to the CPU memory
    err = queue.enqueueReadBuffer(bufC, CL_TRUE, 0, M*N*sizeof(*C), C, NULL, NULL);
}

void im2col_gpu(const float *im, int channels, int height, int width,
         int ksize, int stride, int pad, float *data_col){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;

    // Configure clBlas
    auto &instance = Setup::getInstance();

    int err;

    auto &ctx = instance.getCtx();
    auto &queue = instance.getQueue();

    int K = channels * ksize * ksize;
    int N = width * height;;

    // Prepare OpenCL memory objects
    cl::Buffer input(ctx, CL_MEM_READ_ONLY, N * channels * sizeof(*im), NULL, &err);
    cl::Buffer output(ctx, CL_MEM_READ_WRITE, N * K * sizeof(*data_col), NULL, &err);

    // Copy matrices to the GPU (also C to erase the results of the previous run)
    err = queue.enqueueWriteBuffer(input, CL_TRUE, 0, N * channels * sizeof(*im), im, NULL, NULL);
    err = queue.enqueueWriteBuffer(output, CL_TRUE, 0, N * K * sizeof(*data_col), data_col, NULL, NULL);

    cl::Kernel im2col = cl::Kernel(instance.getProgram(), "im2col");
    im2col.setArg(0, num_kernels);
    im2col.setArg(1, input);
    im2col.setArg(2, height);
    im2col.setArg(3, width);
    im2col.setArg(4, ksize);
    im2col.setArg(5, pad);
    im2col.setArg(6, stride);
    im2col.setArg(7, height_col);
    im2col.setArg(8, width_col);
    im2col.setArg(9, output);

    #define BLOCK 512
    queue.enqueueNDRangeKernel(im2col, cl::NullRange, cl::NDRange((num_kernels + BLOCK -1) / BLOCK), cl::NDRange(BLOCK));

    queue.finish();

    // Copy the output matrix C back to the CPU memory
    err = queue.enqueueReadBuffer(output, CL_TRUE, 0, N * K * sizeof(*data_col), data_col, NULL, NULL);
}

