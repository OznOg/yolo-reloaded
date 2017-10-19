#include "common.h"

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>
#include <iostream>
#include <string>
#include <vector>

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

       // kernel calculates for each element C=A+B
       std::string kernel_code { R"CLC(
                               __kernel void gemm(const int M, const int N, const int K,
                                                     const __global float* A,
                                                     const __global float* B,
                                                     __global float* C) {

                                   // Thread identifiers
                                   const int globalRow = get_global_id(0); // Row ID of C (0..M)
                                   const int globalCol = get_global_id(1); // Col ID of C (0..N)

                                   // Compute a single element (loop over K)
                                   for (int k = 0; k < K; k++) {
                                       C[globalCol + N * globalRow] += A[k + K * globalRow] * B[globalCol + N * k];
                                   }
                               }
                              )CLC"};

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
    gemmm.setArg(0, M);
    gemmm.setArg(1, N);
    gemmm.setArg(2, K);
    gemmm.setArg(3,bufA);
    gemmm.setArg(4,bufB);
    gemmm.setArg(5,bufC);
    queue.enqueueNDRangeKernel(gemmm, cl::NullRange, cl::NDRange(M, N), cl::NullRange);
    queue.finish();

    // Copy the output matrix C back to the CPU memory
    err = queue.enqueueReadBuffer(bufC, CL_TRUE, 0, M*N*sizeof(*C), C, NULL, NULL);
}

