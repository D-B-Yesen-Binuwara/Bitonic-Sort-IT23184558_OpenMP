/* bitonicMPI_fixed.c   
   Compile: mpicc -O2 bitonicMPI_fixed.c -o bitonicMPI_fixed
*/

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <mpi.h>

// Local bitonic sort functions
static inline void swap_int(int *a, int *b) {
    int t = *a; *a = *b; *b = t;
}

// Compare and swap elements
void bitonic_compare_and_swap(int arr[], int low, int k, int dir) {
    for (int i = low; i < low + k; ++i) {
        if (dir) { // ascending
            if (arr[i] > arr[i + k]) swap_int(&arr[i], &arr[i + k]);
        } else {   // descending
            if (arr[i] < arr[i + k]) swap_int(&arr[i], &arr[i + k]);
        }
    }
}

void bitonic_merge_recursive(int arr[], int low, int cnt, int dir) {
    if (cnt <= 1) return;
    int k = cnt / 2;
    bitonic_compare_and_swap(arr, low, k, dir);
    bitonic_merge_recursive(arr, low, k, dir);
    bitonic_merge_recursive(arr, low + k, k, dir);
}

void bitonic_sort_recursive(int arr[], int low, int cnt, int dir) {
    if (cnt <= 1) return;
    int k = cnt / 2;
    bitonic_sort_recursive(arr, low, k, 1);
    bitonic_sort_recursive(arr, low + k, k, 0);
    bitonic_merge_recursive(arr, low, cnt, dir);
}

// Helper functions
int next_power_of_two(int n) {
    if (n <= 0) return 1;
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

int is_power_of_two(int x) {
    return x > 0 && ( (x & (x - 1)) == 0 );
}

// Merge two sorted arrays and keep either smaller or larger half
void merge_and_select(const int *a, const int *b, int *dst, int len, int keep_low) {
    int *tmp = (int*)malloc(sizeof(int) * 2 * len);
    if (!tmp) { perror("malloc tmp"); MPI_Abort(MPI_COMM_WORLD, 1); }
    int i = 0, j = 0, t = 0;
    while (i < len && j < len) {
        if (a[i] <= b[j]) tmp[t++] = a[i++];
        else tmp[t++] = b[j++];
    }
    while (i < len) tmp[t++] = a[i++];
    while (j < len) tmp[t++] = b[j++];
    if (keep_low) {
        memcpy(dst, tmp, sizeof(int) * len);
    } else {
        memcpy(dst, tmp + len, sizeof(int) * len);
    }
    free(tmp);
}

// Check if array is sorted
int verify_sorted(const int *global, int n) {
    for (int i = 1; i < n; ++i) if (global[i-1] > global[i]) return 0;
    return 1;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2 && rank == 0) {
        printf("Usage: %s <n>\n", argv[0]);
    }
    int n = 1024;
    if (argc > 1) n = atoi(argv[1]);
    if (n <= 0) n = 1024;

    // Need power of 2 processes
    if (!is_power_of_two(size)) {
        if (rank == 0) fprintf(stderr, "ERROR: number of processes (P=%d) must be a power of two.\n", size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Pad array size to work with process count
    int N = next_power_of_two(n);
    while (N % size != 0) N <<= 1; // increase until divisible by processes

    int local_size = N / size;
    if (local_size <= 0) {
        if (rank == 0) fprintf(stderr, "ERROR: local_size <= 0 (N=%d size=%d)\n", N, size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("MPI checked bitonic: requested n=%d padded N=%d processes=%d local_size=%d\n",
               n, N, size, local_size);
    }

    // Memory allocation
    int *global_arr = NULL;
    if (rank == 0) {
        global_arr = (int*)malloc(sizeof(int) * N);
        if (!global_arr) { perror("malloc global_arr"); MPI_Abort(MPI_COMM_WORLD, 1); }
        // Initialize with random data
        srand(42);
        for (int i = 0; i < n; ++i) global_arr[i] = rand() % 1000000;
        for (int i = n; i < N; ++i) global_arr[i] = INT_MAX;
    }

    int *local = (int*)malloc(sizeof(int) * local_size);
    if (!local) { perror("malloc local"); MPI_Abort(MPI_COMM_WORLD, 1); }

    // MPI: Distribute data chunks to all processes
    MPI_Barrier(MPI_COMM_WORLD); // sync before timing
    double t0 = MPI_Wtime();
    MPI_Scatter(global_arr, local_size, MPI_INT, local, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process sorts its local chunk independently
    bitonic_sort_recursive(local, 0, local_size, 1);

    // Buffers for data exchange
    int *recv_buf = (int*)malloc(sizeof(int) * local_size);
    int *new_local = (int*)malloc(sizeof(int) * local_size);
    if (!recv_buf || !new_local) { perror("malloc buffers"); MPI_Abort(MPI_COMM_WORLD, 1); }

    // MPI: Distributed bitonic network - log(P) phases of partner communication
    for (int k = 2; k <= size; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int partner = rank ^ j; // XOR to find communication partner

            // Determine sort direction based on position in bitonic network
            int ascending_block = ((rank & k) == 0);
            int lower_partner = ((rank & j) == 0);
            int keep_low = ascending_block ? lower_partner : (1 - lower_partner);

            // MPI: Exchange sorted chunks with partner process
            MPI_Sendrecv(local, local_size, MPI_INT, partner, 0,
                         recv_buf, local_size, MPI_INT, partner, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Merge received data and keep smaller/larger half
            merge_and_select(local, recv_buf, new_local, local_size, keep_low);
            memcpy(local, new_local, sizeof(int) * local_size);

            MPI_Barrier(MPI_COMM_WORLD); // sync after each merge step
        }
    }

    // MPI: Gather sorted chunks back to process 0
    MPI_Gather(local, local_size, MPI_INT, global_arr, local_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        double elapsed = t1 - t0;
        printf("Elapsed time: %.6f s\n", elapsed);
        int ok = verify_sorted(global_arr, n);
        printf("Result: %s\n", ok ? "SORTED" : "NOT SORTED");
        if (!ok) {
            fprintf(stderr, "DEBUG: printing first 64 values (padding shown as INT_MAX):\n");
            int end = (n < 64) ? n : 64;
            for (int i = 0; i < end; ++i) {
                if (global_arr[i] == INT_MAX) printf("[PAD] ");
                else printf("%d ", global_arr[i]);
            }
            printf("\n");
        }
        free(global_arr);
    }

    free(local);
    free(recv_buf);
    free(new_local);

    MPI_Finalize(); // cleanup MPI environment
    return 0;
}
