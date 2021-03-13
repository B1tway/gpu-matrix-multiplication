
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <utility>
#include <chrono>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdio.h>
#define BLOCK_SIZE 32
#define ROWS 2048
#define COLUMNS 2048
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
cudaError_t mulWithCuda(float* a, float* b, float* c, int N);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
__global__ void simple_mul(
    float* X,
    float* Y,
    float* S,
    const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float res = 0;
    for (int k = 0; k < N; k++) {
        res += X[j * N + k] * Y[k * N + i];
    }
    S[j * N + i] = res;
}
__global__ void local_mul(float* X, float* Y, float* S,
    const int K) {
    int global_row = blockIdx.x * blockDim.x + threadIdx.x;
    int global_col = blockIdx.y * blockDim.y + threadIdx.y;
    int local_row = threadIdx.x;
    int local_col = threadIdx.y;

    __shared__ float localX[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float localY[BLOCK_SIZE][BLOCK_SIZE];
    float res = 0;
    for (int kg = 0; kg < K / BLOCK_SIZE; kg++) {
        int aid = global_col * K + (kg * BLOCK_SIZE + local_row);
        int bid = (kg * BLOCK_SIZE + local_col) * K + global_row;
        localX[local_col][local_row] = X[aid];
        localY[local_col][local_row] = Y[bid];
        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; i++) {
            res += localX[local_col][i] * localY[i][local_row];
        }
        __syncthreads();
    }

    S[global_col * K + global_row] = res;
}
int main()
{
    float* a = new float[ROWS * COLUMNS];
    int N = ROWS;
    for (size_t i = 0; i < ROWS; i++)
    {
        for (size_t j = 0; j < COLUMNS; j++)
        {
            a[i * COLUMNS + j] = rand() / 1e5;
            //a[i * COLUMNS + j] = 1;

        }
    }
    float* b = new float[ROWS * COLUMNS];
    for (size_t i = 0; i < ROWS; i++)
    {
        for (size_t j = 0; j < COLUMNS; j++)
        {
            b[i * COLUMNS + j] = rand() / 1e5;
            //b[i * COLUMNS + j] = 1;

        }
    }
    float* res = new float[ROWS * COLUMNS];
    for (size_t i = 0; i < ROWS; i++)
    {
        for (size_t j = 0; j < COLUMNS; j++)
        {
            res[i * COLUMNS + j] = 0;
        }
    }

    // Add vectors in parallel.
    cudaError_t cudaStatus = mulWithCuda(a, b, res, N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
 /*   std::cout << "\nMatrix #RESULT: \n";*/
   /* for (size_t i = 0; i < ROWS; i++)
    {
        for (size_t j = 0; j < COLUMNS; j++)
        {
            std::cout << res[i * COLUMNS + j] << " ";
        }
        std::cout << "\n";
    }
   */
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
cudaError_t mulWithCuda(float* a, float* b, float* c, int N)
{
    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
  
    cudaStatus = cudaMalloc((void**)&dev_c, ROWS * COLUMNS * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, ROWS * COLUMNS * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, ROWS * COLUMNS * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_a, a, ROWS * COLUMNS * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, ROWS * COLUMNS * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    unsigned int grid_rows = (ROWS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (COLUMNS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    // Launch a kernel on the GPU with one thread for each element.
    local_mul<<<dimGrid, dimBlock>>> (dev_a, dev_b, dev_c, N);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, ROWS * COLUMNS * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
