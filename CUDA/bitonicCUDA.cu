%%bash
cat > bitonicCUDA02.cu << 'EOF'
/* bitonicCUDA.cu
   Compile: nvcc -O2 bitonicCUDA.cu -o bitonicCUDA
*/

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda_runtime.h>

// CUDA kernel for bitonic compare-swap step
// Each thread finds its XOR partner and compares/swaps
__global__ void bitonic_step(int *arr, int j, int k)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ixj = i ^ j; // XOR to find partner index

    // Only process if we're the lower index to avoid double-swapping
    if (ixj > i)
    {
        // Determine sort direction based on position in bitonic network
        bool ascending = ((i & k) == 0);

        // Compare and swap if needed
        if ((arr[i] > arr[ixj]) == ascending) {
            int temp = arr[i];
            arr[i] = arr[ixj];
            arr[ixj] = temp;
        }
    }
}

// Find next power of 2 for array padding
int next_power_of_two(int n)
{
    if (n <= 1) return 1;
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

// Check if array is properly sorted
int is_sorted(int *arr, int n)
{
    for (int i = 1; i < n; i++)
        if (arr[i - 1] > arr[i])
            return 0;
    return 1;
}


int main(int argc, char *argv[])
{
    int n = 1024;
    int threads_per_block = 256;

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) threads_per_block = atoi(argv[2]);

    if (threads_per_block <= 0 || threads_per_block > 1024) {
        printf("Invalid block size. Using 256.\n");
        threads_per_block = 256;
    }

    int m = next_power_of_two(n); // Pad to power of 2

    printf("CUDA Bitonic Sort (Correct Version)\n");
    printf("Array size: %d (padded to %d), Threads per block: %d\n",
           n, m, threads_per_block);

    // Allocate host memory
    int *h_arr = (int*)malloc(m * sizeof(int));
    if (!h_arr) {
        fprintf(stderr, "Host malloc failed.\n");
        return 1;
    }

    // Fill array with random data
    srand(42);
    for (int i = 0; i < n; i++)
        h_arr[i] = rand() % 10000;

    for (int i = n; i < m; i++)
        h_arr[i] = INT_MAX; // Pad with large values

    // CUDA: Allocate GPU memory and transfer data once
    int *d_arr;
    cudaMalloc(&d_arr, m * sizeof(int));
    cudaMemcpy(d_arr, h_arr, m * sizeof(int), cudaMemcpyHostToDevice);

    // CUDA timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // CUDA: Main bitonic sort network - runs entirely on GPU
    for (int k = 2; k <= m; k <<= 1) {        // sequence length
        for (int j = k >> 1; j > 0; j >>= 1) { // comparison distance
            int total_threads = m;
            int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

            // Launch kernel with thousands of GPU threads
            bitonic_step<<<blocks, threads_per_block>>>(d_arr, j, k);
            cudaDeviceSynchronize(); // wait for completion
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // CUDA: Copy sorted result back to host
    cudaMemcpy(h_arr, d_arr, m * sizeof(int), cudaMemcpyDeviceToHost);

    // Display results
    printf("Execution time: %.6f seconds\n", ms / 1000.0);
    printf("Result: %s\n", is_sorted(h_arr, n) ? "SORTED" : "NOT SORTED");

    // Clean up memory
    cudaFree(d_arr);
    free(h_arr);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
EOF