#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void my_dgesv(int N, double *A, double *B) {
    int i, j, k;
    double pivot;
    int *pivots = (int *)malloc(N * sizeof(int));
    double *P = (double *)malloc(N * N * sizeof(double));

    // Initialize pivot array and the permutation matrix P
    for (i = 0; i < N; i++) {
        pivots[i] = i ; // Start from 1, not 0
        for (j = 0; j < N; j++) {
            if (i == j) {
                P[i * N + j] = 1.0; // Diagonal elements of P are 1
            } else {
                P[i * N + j] = 0.0;
            }
        }
    }

    // Forward elimination with partial pivoting
    for (i = 0; i < N; i++) {
        // Find the pivot element with the largest magnitude
        pivot = fabs(A[(pivots[i] - 1) * N + i]);
        int pivot_row = i;

        for (j = i + 1; j < N; j++) {
            double abs_val = fabs(A[(pivots[j] - 1) * N + i]);
            if (abs_val > pivot) {
                pivot = abs_val;
                pivot_row = j;
            }
        }

        // Swap row indices in the pivot array
        int temp = pivots[i];
        pivots[i] = pivots[pivot_row];
        pivots[pivot_row] = temp;

        // Update permutation matrix P accordingly
        for (j = 0; j < N; j++) {
            double temp_p = P[i * N + j];
            P[i * N + j] = P[pivot_row * N + j];
            P[pivot_row * N + j] = temp_p;
        }

        // Divide the current row by the pivot element
        pivot = A[(pivots[i] - 1) * N + i];
        for (j = i; j < N; j++) {
            A[(pivots[i] - 1) * N + j] /= pivot;
        }
        for (j = 0; j < N; j++) {
            B[(pivots[i] - 1) * N + j] /= pivot;
        }

        // Eliminate other rows
        for (k = i + 1; k < N; k++) {
            pivot = A[(pivots[k] - 1) * N + i];
            for (j = i; j < N; j++) {
                A[(pivots[k] - 1) * N + j] -= pivot * A[(pivots[i] - 1) * N + j];
            }
            for (j = 0; j < N; j++) {
                B[(pivots[k] - 1) * N + j] -= pivot * B[(pivots[i] - 1) * N + j];
            }
        }
    }

    // Back substitution
    for (i = N - 1; i >= 0; i--) {
        for (k = i - 1; k >= 0; k--) {
            pivot = A[(pivots[k] - 1) * N + i];
            for (j = i; j < N; j++) {
                A[(pivots[k] - 1) * N + j] -= pivot * A[(pivots[i] - 1) * N + j];
            }
            for (j = 0; j < N; j++) {
                B[(pivots[k] - 1) * N + j] -= pivot * B[(pivots[i] - 1) * N + j];
            }
        }
    }

    // Apply permutation matrix P to the solution B
    double *temp_B = (double *)malloc(N * N * sizeof(double));
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            temp_B[i * N + j] = B[(pivots[i] - 1) * N + j];
        }
    }

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            B[i * N + j] = 0.0;
            for (k = 0; k < N; k++) {
                B[i * N + j] += P[i * N + k] * temp_B[k * N + j];
            }
        }
    }

    // Clean up
    free(pivots);
    free(P);
    free(temp_B);
}
