/* bitonicOmp02.c
   Compile: gcc -fopenmp -O2 bitonicOmp02.c -o bitonicOmp02
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

// Bitonic merge with task-based parallelism
void bitonic_merge(int arr[], int low, int cnt, int dir) {
    if (cnt > 1) {
        int k = cnt / 2;

        // Parallel compare-exchange operations
        #pragma omp parallel for schedule(static) if(k > 1000)
        for (int i = low; i < low + k; i++) {
            if (dir == 1) {                // ascending
                if (arr[i] > arr[i + k]) swap_int(&arr[i], &arr[i + k]);
            } else {                       // descending
                if (arr[i] < arr[i + k]) swap_int(&arr[i], &arr[i + k]);
            }
        }

        // Create tasks for recursive calls (only for large chunks)
        // Tasks allow dynamic work distribution among all threads
        if (k > 2048) {
            #pragma omp task
            bitonic_merge(arr, low, k, dir);
            
            #pragma omp task
            bitonic_merge(arr, low + k, k, dir);
            
            #pragma omp taskwait
        } else {
            // Run sequentially for small chunks
            bitonic_merge(arr, low, k, dir);
            bitonic_merge(arr, low + k, k, dir);
        }
    }
}

// Bitonic sort with task-based parallelism
void bitonic_sort_recursive(int arr[], int low, int cnt, int dir) {
    if (cnt > 1) {
        int k = cnt / 2;

        // Create tasks for recursive calls
        if (k > 2048) {
            #pragma omp task
            bitonic_sort_recursive(arr, low, k, 1);      // 1st half ascending

            #pragma omp task
            bitonic_sort_recursive(arr, low + k, k, 0);  // 2nd half descending
            
            #pragma omp taskwait
        } else {
            // Run sequentially for small chunks
            bitonic_sort_recursive(arr, low, k, 1);
            bitonic_sort_recursive(arr, low + k, k, 0);
        }
        
        // Merge both halves
        bitonic_merge(arr, low, cnt, dir);
    }
}

// Initialize parallel region for task execution
void bitonic_sort_parallel(int arr[], int n) {
    #pragma omp parallel
    {
        // One thread creates tasks, others execute them
        #pragma omp single
        {
            bitonic_sort_recursive(arr, 0, n, 1);
        }
    }
}

// Find next power of 2
int next_power_of_two(int n) {
    if (n <= 1) return 1;
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

int main(int argc, char *argv[]) {
    int n = 1024;
    int num_threads = omp_get_max_threads();
    
    if (argc > 1) n = atoi(argv[1]);
    // Set thread count
    if (argc > 2) {
        num_threads = atoi(argv[2]);
        omp_set_num_threads(num_threads);
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

    srand(42);
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 10000;
    }
    for (int i = n; i < m; i++) arr[i] = INT_MAX;

    printf("OpenMP Bitonic Sort (Task-based) - Array size: %d, Threads: %d\n", n, num_threads);
    
    double start_time = omp_get_wtime();
    bitonic_sort_parallel(arr, m);
    double end_time = omp_get_wtime();
    
    double execution_time = end_time - start_time;
    printf("Execution time: %.6f seconds\n", execution_time);
    
    // Check if sorted correctly
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