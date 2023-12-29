#include <openblas/lapacke.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include "timer.h"
#include "dgesv.h"

double *generate_matrix(unsigned int size, unsigned int seed)
{
    unsigned int i;
    double *matrix = (double *)malloc(sizeof(double) * size * size);
    
    if (matrix == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for matrix\n");
        exit(EXIT_FAILURE);
    }

    srand(seed);

    for (i = 0; i < size * size; i++) {
        matrix[i] = rand() % 100;
    }

    return matrix;
}

double *duplicate_matrix(double *orig, unsigned int size)
{
    double *replica = (double *)malloc(sizeof(double) * size * size);

    if (replica == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for replica matrix\n");
        exit(EXIT_FAILURE);
    }

    memcpy((void *)replica, (void *)orig, size * size * sizeof(double));

    return replica;
}

int is_nearly_equal(double x, double y)
{
    const double epsilon = 1e-5 /* some small number */;
    return fabs(x - y) <= epsilon * fabs(x);
    // see Knuth section 4.2.2 pages 217-218
}

unsigned int check_result(double *bref, double *b, unsigned int size)
{
    unsigned int i;

    for (i = 0; i < size * size; i++) {
        if (!is_nearly_equal(bref[i], b[i]))
            return 0;
    }

    return 1;
}

void print_matrix(double *matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}


int main(int argc, char *argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <matrix_size>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int size = atoi(argv[1]);

    if (size < 1) {
        fprintf(stderr, "Error: Matrix size must be at least 1\n");
        exit(EXIT_FAILURE);
    }

    double *a, *aref;
    double *b, *bref;

    a = generate_matrix(size, 1);
    b = generate_matrix(size, 2);
    aref = duplicate_matrix(a, size);
    bref = duplicate_matrix(b, size);

    int n = size, nrhs = size, lda = size, ldb = size, info;
    int *ipiv = (int *)malloc(sizeof(int) * size);

    if (ipiv == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for ipiv array\n");
        exit(EXIT_FAILURE);
    }

    timeinfo start, now;
    timestamp(&start);

    info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, aref, lda, ipiv, bref, ldb);

    if (info != 0) {
        fprintf(stderr, "Error: LAPACKE_dgesv failed with error code %d\n", info);
        exit(EXIT_FAILURE);
    }

    timestamp(&now);
    printf("\nTime taken by Lapacke dgesv: %ld ms\n", diff_milli(&start, &now));

    // printf("\n\nResults from LAPACK:\n");
    // print_matrix(bref, size, size);

    timestamp(&start);

    my_dgesv(n, a, b);

    timestamp(&now);
    printf("\nTime taken by my dgesv solver: %ld ms\n", diff_milli(&start, &now));

    // printf("\n\nResults from my_dgesv:\n");
    // print_matrix(b, size, size);

    if (check_result(bref, b, size) == 1)
        printf("Result is ok!\n");
    else
        printf("\nResult is wrong!\n\n");

    free(ipiv);
    free(a);
    free(b);
    free(aref);
    free(bref);

    return 0;
}
