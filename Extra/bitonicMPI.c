/* bitonicMPI.c
   MPI implementation of bitonic sort using distributed memory parallelization
   
   Parallelization: Each process sorts local data, then performs distributed bitonic merge
   using partner communication in log(P) phases where P = number of processes.
   
   Compile: mpicc -O2 bitonicMPI.c -o bitonicMPI
   Run: mpirun -np <num_processes> ./bitonicMPI
*/

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <mpi.h>

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
        for (int i = low; i < low + k; i++) {
            if (dir == 1) {                // ascending
                if (arr[i] > arr[i + k]) swap_int(&arr[i], &arr[i + k]);
            } else {                       // descending
                if (arr[i] < arr[i + k]) swap_int(&arr[i], &arr[i + k]);
            }
        }
        // Recursively merge both halves in same direction
        bitonic_merge(arr, low, k, dir);
        bitonic_merge(arr, low + k, k, dir);
    }
}

// Bitonic sort: recursively creates bitonic sequences then merges them
// Strategy: Sort first half ascending, second half descending to create bitonic sequence,
// then merge entire sequence in desired direction
void bitonic_sort_recursive(int arr[], int low, int cnt, int dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        // 1st half -> ascending, 2nd half -> descending to form bitonic seq
        bitonic_sort_recursive(arr, low, k, 1);      // ascending
        bitonic_sort_recursive(arr, low + k, k, 0);  // descending
        // merge whole sequence in direction need
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
    int rank, size, n = 1024;
    
    // PARALLELIZATION- Initialize MPI environment and geting process info
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get process ID (0 to size-1)
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get total number of processes

    if (argc > 1) n = atoi(argv[1]);
    if (n <= 0) n = 1024;
    
    // PARALLELIZATION- Broadcast array size to all processes
    // Ensures all processes work with same problem size
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // PARALLELIZATION- Calculate data distribution
    // Each process gets equal portion of padded array
    int total_size = next_power_of_two(n);
    if (total_size < size) total_size = size;  // Ensure at least 1 element per process
    
    int local_size = total_size / size;  // Elements per process
    int *local_arr = malloc(sizeof(int) * local_size);  // Local data for this process
    int *global_arr = NULL;  // Only rank 0 needs global array

    double start_time, end_time;
    // PARALLELIZATION- Only rank 0 initializes data and starts timing
    // avoids duplicate initialization and ensures consistent timing
    if (rank == 0) {
        global_arr = malloc(sizeof(int) * total_size);
        srand(42);  // Fixed seed for reproducible results across runs
        
        // Initialize with random data
        for (int i = 0; i < n; i++) {
            global_arr[i] = rand() % 10000;
        }
        // Pad remaining elements with INT_MAX (will sort to end)
        for (int i = n; i < total_size; i++) {
            global_arr[i] = INT_MAX;
        }
        printf("MPI Bitonic Sort - Array size: %d, Processes: %d\n", n, size);
        start_time = MPI_Wtime();
    }

    // PARALLELIZATION- Distributing data to all processes equally
    // Scatter divides global array into equal chunks for each process
    MPI_Scatter(global_arr, local_size, MPI_INT, local_arr, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process sorts its local data in ascending order
    bitonic_sort_recursive(local_arr, 0, local_size, 1);

    // Distributed bitonic sort network
    for (int k = 2; k <= size; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int partner = rank ^ j;
            
            // Exchange data with partner
            int *recv_arr = malloc(sizeof(int) * local_size);
            MPI_Sendrecv(local_arr, local_size, MPI_INT, partner, 0,
                         recv_arr, local_size, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Determine if we keep min or max elements
            int keep_min = ((rank & k) == 0);
            
            // Compare and exchange elements
            for (int i = 0; i < local_size; i++) {
                if (keep_min) {
                    if (local_arr[i] > recv_arr[i]) {
                        swap_int(&local_arr[i], &recv_arr[i]);
                    }
                } else {
                    if (local_arr[i] < recv_arr[i]) {
                        swap_int(&local_arr[i], &recv_arr[i]);
                    }
                }
            }
            
            free(recv_arr);
        }
    }

    // PARALLELIZATION- Collect sorted data back to rank 0
    // Gather combines all local arrays into final sorted global array
    MPI_Gather(local_arr, local_size, MPI_INT, global_arr, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Only rank 0 performs final verification and timing
    if (rank == 0) {
        end_time = MPI_Wtime();
        double execution_time = end_time - start_time;
        printf("Execution time: %.6f seconds\n", execution_time);
        
        /* Verify that array is correctly sorted */
        int sorted = 1;
        int first_error = -1;
        for (int i = 1; i < n; i++) {
            if (global_arr[i-1] > global_arr[i]) {
                sorted = 0;
                if (first_error == -1) first_error = i;
                break;
            }
        }
        printf("Result: %s", sorted ? "SORTED" : "NOT SORTED");
        if (!sorted) {
            printf(" (first error at index %d: %d > %d)", first_error, global_arr[first_error-1], global_arr[first_error]);
        }
        printf("\n");
        
        // Debug: Print first and last few elements
        printf("First 10: ");
        for (int i = 0; i < 10 && i < n; i++) {
            printf("%d ", global_arr[i]);
        }
        printf("\nLast 10: ");
        for (int i = (n > 10 ? n-10 : 0); i < n; i++) {
            printf("%d ", global_arr[i]);
        }
        printf("\n");
        
        free(global_arr);
    }

    free(local_arr);
    /* PARALLELIZATION: Clean up MPI environment */
    MPI_Finalize();
    return 0;
}