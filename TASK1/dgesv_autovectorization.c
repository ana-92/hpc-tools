void my_dgesv(int N, double *A, double *B) {
    int i, j, k;
    double pivot;
    int *pivots = (int *)malloc(N * sizeof(int));

    if (pivots == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for pivots array\n");
        exit(EXIT_FAILURE);
    }

    // Key Modification 1: Initialize pivots with consecutive values
    for (i = 0; i < N; i++) {
        pivots[i] = i;
    }

    for (i = 0; i < N; i++) {
        pivot = fabs(A[pivots[i] * N + i]);
        int pivot_row = i;

        for (j = i + 1; j < N; j++) {
            double abs_val = fabs(A[pivots[j] * N + i]);
            if (abs_val > pivot) {
                pivot = abs_val;
                pivot_row = j;
            }
        }

        int temp = pivots[i];
        pivots[i] = pivots[pivot_row];
        pivots[pivot_row] = temp;

        // Key Modification 2: Loop Restructuring for vectorization
        for (j = i; j < N; j++) {
            A[pivots[i] * N + j] /= pivot;
        }

        for (j = 0; j < N; j++) {
            B[pivots[i] * N + j] /= pivot;
        }

        for (k = 0; k < N; k++) {
            if (k != i) {
                pivot = A[pivots[k] * N + i];
                // Key Modification 2: Loop Restructuring for vectorization
                for (j = i; j < N; j++) {
                    A[pivots[k] * N + j] -= pivot * A[pivots[i] * N + j];
                }
                for (j = 0; j < N; j++) {
                    B[pivots[k] * N + j] -= pivot * B[pivots[i] * N + j];
                }
            }
        }
    }

    // Key Modification 3: Utilize memcpy for copying
    double *tempB = (double *)malloc(N * N * sizeof(double));
    if (tempB == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for tempB array\n");
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < N; i++) {
        memcpy(&tempB[i * N], &B[pivots[i] * N], N * sizeof(double));
    }

    memcpy(B, tempB, N * N * sizeof(double));

    free(tempB);
    free(pivots);
}
