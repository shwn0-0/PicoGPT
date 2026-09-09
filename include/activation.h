#ifndef ACTIVATION_H
#define ACTIVATION_H
#include "matrix.h"
#define EPS 1e-8

typedef struct NNActivationLayer {
  void (*forward)(struct NNActivationLayer *layer, const Matrix *in);
  void (*backward)(struct NNActivationLayer *layer, const Matrix *in, const Matrix* loss);
  Matrix* loss;
  Matrix* output;
} NNActivationLayer;
#endif
