#ifndef MATRIX_H
#define MATRIX_H
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct Matrix {
  size_t rows;
  size_t cols;
  float *data;
} Matrix;

Matrix *newMatrix(size_t rows, size_t cols);
void deleteMatrix(Matrix *m);

void setMatrixValue(Matrix *m, size_t r, size_t c, float val);
float getMatrixValue(const Matrix *m, size_t r, size_t c);

void initMatrix(Matrix *m, float vals[]);
void randomMatrix(Matrix *m, float min, float max);
void fillMatrix(Matrix *m, float fill);
void identityMatrix(Matrix *m, float scale);

void displayMatrix(const char *title, const Matrix *m);

float matrixSum(const Matrix *in);
bool matrixAdd(const Matrix *a, const Matrix *b, Matrix *out);
bool matrixMul(const Matrix *a, const Matrix *b, Matrix *out);
bool matrixScale(const Matrix *m, float scale, Matrix *out);
#endif
