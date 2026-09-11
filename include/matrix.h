#ifndef MATRIX_H
#define MATRIX_H
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct Vec3 {
  size_t x;
  size_t y;
  size_t z;
  size_t w;
} MatrixDim, Vec4;

typedef struct Matrix {
  MatrixDim dim;
  float *data;
} Matrix;

Matrix *newMatrix(MatrixDim dim);
void deleteMatrix(Matrix *m);

void setMatrixValue(Matrix *m, Vec4 idx, float val);
float getMatrixValue(const Matrix *m, Vec4);

void initMatrix(Matrix *m, float vals[]);
void randomMatrix(Matrix *m, float min, float max);
void fillMatrix(Matrix *m, float fill);

void displayMatrix(const char *title, const Matrix *m);

size_t matrixSize(const Matrix *m);
float matrixSum(const Matrix *in);
bool matrixAdd(const Matrix *a, const Matrix *b, Matrix *out);
bool matrixMul(const Matrix *a, const Matrix *b, Matrix *out);
bool matrixScale(const Matrix *m, float scale, Matrix *out);
#endif
