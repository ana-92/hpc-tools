#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

void my_dgesv(int N, double *A, double *B) {
    int i, j, k;
    double pivot;
    int *pivots = (int *)malloc(N * sizeof(int));

    if (pivots == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for pivots array\n");
        exit(EXIT_FAILURE);
    }

    #pragma omp parallel for
    for (i = 0; i < N; i++) {
        pivots[i] = i;
    }

    for (i = 0; i < N; i++) {
        pivot = fabs(A[pivots[i] * N + i]);
        int pivot_row = i;

        #pragma omp parallel for reduction(min: pivot) private(j)
        for (j = i + 1; j < N; j++) {
            double abs_val = fabs(A[pivots[j] * N + i]);
            if (abs_val > pivot) {
                #pragma omp critical
                {
                    if (abs_val > pivot) {
                        pivot = abs_val;
                        pivot_row = j;
                    }
                }
            }
        }

        int temp = pivots[i];
        pivots[i] = pivots[pivot_row];
        pivots[pivot_row] = temp;

        #pragma omp parallel for
        for (j = 0; j < N; j++) {
            double temp_a = A[pivots[i] * N + j];
            A[pivots[i] * N + j] = A[pivots[pivot_row] * N + j];
            A[pivots[pivot_row] * N + j] = temp_a;

            double temp_b = B[pivots[i] * N + j];
            B[pivots[i] * N + j] = B[pivots[pivot_row] * N + j];
            B[pivots[pivot_row] * N + j] = temp_b;
        }

        pivot = A[pivots[i] * N + i];

        #pragma omp parallel for
        for (j = i; j < N; j++) {
            A[pivots[i] * N + j] /= pivot;
            B[pivots[i] * N + j] /= pivot;
        }

        #pragma omp parallel for private(k, pivot)
        for (k = 0; k < N; k++) {
            if (k != i) {
                pivot = A[pivots[k] * N + i];
                #pragma omp parallel for
                for (j = i; j < N; j++) {
                    A[pivots[k] * N + j] -= pivot * A[pivots[i] * N + j];
                }
                #pragma omp parallel for
                for (j = 0; j < N; j++) {
                    B[pivots[k] * N + j] -= pivot * B[pivots[i] * N + j];
                }
            }
        }
    }

    double *tempB = (double *)malloc(N * N * sizeof(double));
    if (tempB == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for tempB array\n");
        exit(EXIT_FAILURE);
    }

    #pragma omp parallel for
    for (i = 0; i < N; i++) {
        #pragma omp parallel for
        for (j = 0; j < N; j++) {
            tempB[i * N + j] = B[pivots[i] * N + j];
        }
    }

    memcpy(B, tempB, N * N * sizeof(double));

    free(tempB);
    free(pivots);
}
