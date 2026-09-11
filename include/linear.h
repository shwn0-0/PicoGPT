#ifndef LINEAR_H
#define LINEAR_H

#include "matrix.h"


typedef struct NNLinearLayer {
  size_t inputDim;
  size_t outputDim;
  Matrix *weights;
  Matrix *weightsGrad;
  Matrix *prevWeightsGrad;
  Matrix *bias;
  Matrix *biasGrad;
  Matrix *prevBiasGrad;
  Matrix *output;
  Matrix *loss;
  void (*forward)(struct NNLinearLayer *nnLayer, Matrix *input);
  void (*backward)(struct NNLinearLayer *nnLayer, Matrix *input, Matrix *loss);
  void (*optimize)(struct NNLinearLayer *nnLayer, float trainingSteps,
                                 float learningRate, float decay);
} NNLinearLayer;

NNLinearLayer *newLinearLayer(size_t inputDim, size_t outputDim);
void initLinearLayer(NNLinearLayer *nnLayer, float min, float max);
void destroyLinearLayer(NNLinearLayer *layer);
#endif
