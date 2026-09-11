#ifndef ACTIVATION_H
#define ACTIVATION_H
#include "matrix.h"

typedef struct NNActivationLayer {
  void (*forward)(struct NNActivationLayer *layer, const Matrix *in);
  void (*backward)(struct NNActivationLayer *layer, const Matrix *in, const Matrix* loss);
  Matrix* loss;
  Matrix* output;
} NNActivationLayer;

void deleteActivationLayer(NNActivationLayer *layer);
#endif
