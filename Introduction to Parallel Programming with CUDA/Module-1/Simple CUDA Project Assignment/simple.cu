#include "simple.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Device function: called from GPU
__device__ float deviceMultiply(float a, float b)
{
    return a * b;
}

// Kernel: runs on GPU
__global__ void vectorMult(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        C[i] = deviceMultiply(A[i], B[i]);
    }
}

// Host main function
int main()
{
    const int numElements = N;
    size_t size = numElements * sizeof(float);

    printf("[Vector multiplication of %d elements]\n", numElements);

    // Host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    if (!h_A || !h_B || !h_C)
    {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(1);
    }

    // Initialize
    for (int i = 0; i < numElements; i++)
    {
        h_A[i] = (float)i;
        h_B[i] = (float)(i * 2);  // so C[i] = i * 2*i = 2*i^2
    }

    // Device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Kernel launch config
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    // Launch kernel
    vectorMult<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // Wait for completion
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print first 10 results
    for (int i = 0; i < 10 && i < numElements; i++)
    {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    // Verify
    bool passed = true;
    for (int i = 0; i < numElements; i++)
    {
        float expected = h_A[i] * h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5)
        {
            printf("Error at %d: got %f, expected %f\n", i, h_C[i], expected);
            passed = false;
        }
    }

    if (passed)
        printf("Test PASSED\n");
    else
        printf("Test FAILED\n");

    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaDeviceReset();

    printf("Done\n");
    return 0;
}