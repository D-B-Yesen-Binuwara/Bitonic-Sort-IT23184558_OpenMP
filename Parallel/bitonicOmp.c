/* bitonicOmp.c
   Serial bitonic sort (recursive), pads to next power of two using INT_MAX.
   Compile: gcc -O2 bitonicOmp.c -o bitonicOmp
*/

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <omp.h>

static inline void swap_int(int *a, int *b) {
    int t = *a;
    *a = *b;
    *b = t;
}

/* direction: 1 for ascending, 0 for descending */
void bitonic_merge(int arr[], int low, int cnt, int dir) {
    if (cnt > 1) {
        int k = cnt / 2;

        #pragma omp parallel for schedule(static)
        for (int i = low; i < low + k; i++) {
            if (dir == 1) {                
                if (arr[i] > arr[i + k]) swap_int(&arr[i], &arr[i + k]);
            } else {
                if (arr[i] < arr[i + k]) swap_int(&arr[i], &arr[i + k]);
            }
        }

        bitonic_merge(arr, low, k, dir);
        bitonic_merge(arr, low + k, k, dir);
    }
}

void bitonic_sort_recursive(int arr[], int low, int cnt, int dir) {
    if (cnt > 1) {
        int k = cnt / 2;

        #pragma omp parallel sections // PARALLELIZE TOP-LEVEL RECURSION ONLY
        {
            #pragma omp section
            bitonic_sort_recursive(arr, low, k, 1);

            #pragma omp section
            bitonic_sort_recursive(arr, low + k, k, 0);
        }

        bitonic_merge(arr, low, cnt, dir);
    }
}

/* returns next power of two >= n */
int next_power_of_two(int n) {
    if (n <= 1) return 1;
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

int main(void) {
    int n;
    char line[64];

    printf("Enter number of elements (or press Enter to use 16 random elements): ");
    if (fgets(line, sizeof(line), stdin) == NULL) return 1;
    if (sscanf(line, "%d", &n) != 1) n = 16;

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

    srand((unsigned)time(NULL));
    printf("Input array (%d elements):\n", n);
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;
        printf("%d ", arr[i]);
    }
    for (int i = n; i < m; i++) arr[i] = INT_MAX;
    printf("\n");

    bitonic_sort_recursive(arr, 0, m, 1);

    printf("Sorted array (first %d elements):\n", n);
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);
    printf("\n");

    free(arr);
    return 0;
}

