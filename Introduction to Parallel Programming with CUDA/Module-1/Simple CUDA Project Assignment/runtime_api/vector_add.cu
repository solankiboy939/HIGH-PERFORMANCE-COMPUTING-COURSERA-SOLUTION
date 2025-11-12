#include <stdio.h>

__global__ void add(int *a, int *b, int *c) {
    int tid = blockIdx.x;
    c[tid] = a[tid] + b[tid];
}

int main() {
    const int N = 100;
    size_t size = N * sizeof(int);

    // Host arrays
    int h_a[N], h_b[N], h_c[N];

    // Initialize
    for (int i = 0; i < N; i++) {
        h_a[i] = N - i;
        h_b[i] = i * i;
    }

    // Device pointers
    int *d_a, *d_b, *d_c;

    // Allocate GPU memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy to GPU
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch kernel: 100 blocks, 1 thread each
    add<<<N, 1>>>(d_a, d_b, d_c);

    // Copy result back
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Print first few results
    printf("Runtime API result (first 5): %d %d %d %d %d\n",
           h_c[0], h_c[1], h_c[2], h_c[3], h_c[4]);

    // Verify
    bool passed = true;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("Error at index %d\n", i);
            passed = false;
        }
    }

    if (passed)
        printf("✅ Runtime API test passed!\n");

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}