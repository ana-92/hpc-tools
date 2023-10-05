#include "dgesv.h"
#include <stdio.h>
#include <stdlib.h>

double my_dgesv(int N, double *A, double *B) {
    int i, j, k;
    double factor;
    double *X = (double *)malloc(N * sizeof(double));

    // Forward elimination
    for (i = 0; i < N - 1; i++) {
        for (k = i + 1; k < N; k++) {
            factor = A[k] / A[i];
            for (j = i; j < N; j++) {
                A[k] -= factor * A[i];
            }
            B[k] -= factor * B[i];
        }
    }

    // Back substitution
    X[N - 1] = B[N - 1] / A[N - 1];
    for (i = N - 2; i >= 0; i--) {
        double sum = 0.0;
        for (j = i + 1; j < N; j++) {
            sum += A[i] * X[j];
        }
        X[i] = (B[i] - sum) / A[i];
    }

    double result = X[0]; // Assuming a single-variable system
    free(X);
    return result;
}
