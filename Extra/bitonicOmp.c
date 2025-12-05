/* bitonicOmp.c
   OpenMP parallel implementation of bitonic sort using shared memory parallelization
      
   Parallelization: Uses fork-join model with parallel sections for recursive calls
   and parallel loops for merge operations. Shared memory eliminates data transfer overhead.
   
   Compile: gcc -fopenmp -O2 bitonicOmp.c -o bitonicOmp
*/

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <omp.h>
#include <sys/time.h>

/* swap two integers */
static inline void swap_int(int *a, int *b) {
    int t = *a;
    *a = *b;
    *b = t;
}

// Bitonic merge: converts a bitonic sequence into monotonic sequence
// compares and swaps elements at distance k apart, then recursively
// direction: 1 for ascending, 0 for descending
void bitonic_merge(int arr[], int low, int cnt, int dir) {
    if (cnt > 1) {
        int k = cnt / 2;

        // PARALLELIZATION- Parallel loop for compare-exchange operations
        // Static scheduling ensures predictable work distrbution among threads
        // Each iteration is independent
        #pragma omp parallel for schedule(static)
        for (int i = low; i < low + k; i++) {
            if (dir == 1) {                // ascending
                if (arr[i] > arr[i + k]) swap_int(&arr[i], &arr[i + k]);
            } else {                       // descending
                if (arr[i] < arr[i + k]) swap_int(&arr[i], &arr[i + k]);
            }
        }

        // Recursive calls - sequential to avoid excessive thread creation.
        bitonic_merge(arr, low, k, dir);
        bitonic_merge(arr, low + k, k, dir);
    }
}

// Bitonic sort: recursively creates bitonic sequences then merges them
// Strategy: Sort first half ascending, second half descending to create bitonic sequence,
void bitonic_sort_recursive(int arr[], int low, int cnt, int dir) {
    if (cnt > 1) {
        int k = cnt / 2;

        // PARALLELIZATION- sections for independent recursive calls
        // sections execute simultaneously on different threads
        #pragma omp parallel sections
        {
            #pragma omp section
            bitonic_sort_recursive(arr, low, k, 1);      // 1st half ascending

            #pragma omp section
            bitonic_sort_recursive(arr, low + k, k, 0);  // 2nd half descending
        }
        // Implicit barrier: all sections gets complete before merge
        bitonic_merge(arr, low, cnt, dir);
    }
}

// Bitonic sort requires array size to be power of 2
// This finds the smallest power of 2 >= n
int next_power_of_two(int n) {
    if (n <= 1) return 1;
    int p = 1;
    while (p < n) p <<= 1;  // Left shift is equivalent to p *= 2
    return p;
}

int main(int argc, char *argv[]) {
    int n = 1024;
    int num_threads = omp_get_max_threads();
    
    if (argc > 1) n = atoi(argv[1]);
    // PARALLELIZATION- Configure OpenMP thread count
    // set via command line or OMP_NUM_THREADS environment variable
    if (argc > 2) {
        num_threads = atoi(argv[2]);
        omp_set_num_threads(num_threads);  // Override default thread count
    }
    
    if (n <= 0) {
        printf("Number of elements must be positive.\n");
        return 1;
    }

    int m = next_power_of_two(n);
    int *arr = malloc(sizeof(int) * m);
    if (!arr) {
        perror("malloc");
        return 1;
    }

    srand(42);  // Fixed seed for reproducible results across runs
    // Initialize with random data
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 10000;
    }
    // Pad remaining elements with INT_MAX (will sort to end)
    for (int i = n; i < m; i++) arr[i] = INT_MAX;

    printf("OpenMP Bitonic Sort - Array size: %d, Threads: %d\n", n, num_threads);
    
    // PARALLELIZATION- Using OpenMP wall-clock timer for accurate parallel timing
    double start_time = omp_get_wtime();
    bitonic_sort_recursive(arr, 0, m, 1);  // Sort entire array ascending
    double end_time = omp_get_wtime();
    
    double execution_time = end_time - start_time;
    printf("Execution time: %.6f seconds\n", execution_time);
    
    int sorted = 1;
    for (int i = 1; i < n; i++) {
        if (arr[i-1] > arr[i]) {
            sorted = 0;
            break;
        }
    }
    printf("Result: %s\n", sorted ? "SORTED" : "NOT SORTED");

    free(arr);
    return 0;
}

