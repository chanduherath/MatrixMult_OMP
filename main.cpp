#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <stdio.h>
#include <stdint.h>
#include <fstream>
using namespace std;

int matrixDimension = 3;
int programmeCategory;

double getRandomDouble(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void createMatrix(double **matrix, int dim){
    for(int i=0; i<dim; i++){
        for(int j=0; j<dim; j++){
            matrix[i][j] = getRandomDouble(0,1000);
        }
    }
}

void sequentialMultiplication(double **matrix1, double **matrix2, double **matrix3, int dim){
    for(int i=0; i<dim; i++){
        for(int j=0; j<dim; j++){
            matrix3[i][j] = 0;
            for(int k=0; k<dim; k++){
                matrix3[i][j] = matrix3[i][j] + (matrix1[i][k]) * (matrix2[k][j]);
            }
        }
    }
}

void parallelMultiplication(double **matrix1, double **matrix2, double **matrix3, int dim){
#pragma omp parallel for
    for(int i=0; i<dim; i++){
#pragma omp parallel for
        for(int j=0; j<dim; j++){
            matrix3[i][j] = 0;
            for(int k=0; k<dim; k++){
                matrix3[i][j] = matrix3[i][j] + (matrix1[i][k]) * (matrix2[k][j]);
            }
        }
    }
}

void optimizedParallelMultiplication(double **matrix1, double **matrix2, double **matrix3,
                                     double **matrix_transpose, int dim){
    for(int_fast16_t i =0; i<dim; i++){//getting the transpose of the second matrix
        for(int_fast16_t j=0; j<dim; j+=5){//loop unrolling
            matrix_transpose[i][j] = matrix2[j][i];
            matrix_transpose[i][j+1] = matrix2[j+1][i];
            matrix_transpose[i][j+2] = matrix2[j+2][i];
            matrix_transpose[i][j+3] = matrix2[j+3][i];
            matrix_transpose[i][j+4] = matrix2[j+4][i];
        }
    }

#pragma omp parallel for
    for(int_fast16_t i=0; i<dim; i++){
        for(int_fast16_t j=0; j<dim; j++){//use of int_fast16_t
            double sum = 0;//tempory variable to store the result
            for(int_fast16_t k=0; k<dim; k+=5){//loop unrolling
                sum = sum + (matrix1[i][k]) * (matrix_transpose[j][k])+
                      (matrix1[i][k+1]) * (matrix_transpose[j][k+1])+
                      (matrix1[i][k+2]) * (matrix_transpose[j][k+2])+
                      (matrix1[i][k+3]) * (matrix_transpose[j][k+3])+
                      (matrix1[i][k+4]) * (matrix_transpose[j][k+4]);
            }
            matrix3[i][j] = sum;
        }
    }
}

void printMatrix(double **matrix, int dim){
    for(int i = 0; i<dim; i++){
        for(int j=0; j<dim; j++){
            printf("%1f ", matrix[i][j]);
        }
        printf("\n");
    }
}


int main(int argc, char* argv[]) {
    std::cout << "Lab 3 & 4!" << std::endl;

    programmeCategory = atoi(argv[1]);// "0" for serial, "1" for loop parallel, "2" for optimized

    for (int matrixDimension = 200; matrixDimension <= 2000; matrixDimension = matrixDimension + 200) {

        double start_s = 0;//for calculate the execution time
        double stop_s = 0;
        double ellapsed_s = 0;

        double **matrix1;
        double **matrix2;
        double **matrix3;//result matrix
        double **matrix_transpose;
        matrix1 = new double *[matrixDimension];
        matrix2 = new double *[matrixDimension];
        matrix3 = new double *[matrixDimension];
        matrix_transpose = new double *[matrixDimension];
        //dynamically allocate an array
        for (int i = 0; i < matrixDimension; i++) {
            matrix1[i] = new double[matrixDimension];
        }
        for (int i = 0; i < matrixDimension; i++) {
            matrix2[i] = new double[matrixDimension];
        }
        for (int i = 0; i < matrixDimension; i++) {
            matrix3[i] = new double[matrixDimension];
        }
        for (int i = 0; i < matrixDimension; i++) {
            matrix_transpose[i] = new double[matrixDimension];
        }


        createMatrix(matrix1, matrixDimension);
        createMatrix(matrix2, matrixDimension);


        if (programmeCategory == 0) {//sequential programme
            start_s = omp_get_wtime();
            sequentialMultiplication(matrix1, matrix2, matrix3, matrixDimension);
            stop_s = omp_get_wtime();
            ellapsed_s = (stop_s - start_s) * 1000;
            printf("Seq Execution Time for dimension %d : %1f \n", matrixDimension, ellapsed_s);

        } else if (programmeCategory == 1) {//multiplication with parallel for loops
            start_s = omp_get_wtime();
            parallelMultiplication(matrix1, matrix2, matrix3, matrixDimension);
            stop_s = omp_get_wtime();
            ellapsed_s = (stop_s - start_s) * 1000;
            printf("Parallel Execution Time for dimension %d : %1f \n", matrixDimension, ellapsed_s);

        } else if (programmeCategory == 2) {//optimized multiplication with parallel for loops
            start_s = omp_get_wtime();
            optimizedParallelMultiplication(matrix1, matrix2, matrix3, matrix_transpose, matrixDimension);
            stop_s = omp_get_wtime();
            ellapsed_s = (stop_s - start_s) * 1000;
            printf("Optimized Parallel Execution Time for dimension %d : %1f \n", matrixDimension, ellapsed_s);

        }

//            free allocated memmory
        for (int i = 0; i < matrixDimension; i++) {
            delete matrix1[i];
        }
        for (int i = 0; i < matrixDimension; i++) {
            delete matrix2[i];
        }
        for (int i = 0; i < matrixDimension; i++) {
            delete matrix3[i];
        }
        for (int i = 0; i < matrixDimension; i++) {
            delete matrix_transpose[i];
        }
        delete(matrix1);
        delete(matrix2);
        delete(matrix3);
        delete(matrix_transpose);

    }
    return 0;
}