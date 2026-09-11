#include "linear.h"

void nnLinearLayerForward(NNLinearLayer *layer, Matrix *input) {
  matrixMul(layer->weights, input, layer->output);
  matrixAdd(layer->bias, layer->output, layer->output);
}

void nnLinearLayerBackward(NNLinearLayer *layer, Matrix *input, Matrix *loss) {
  Matrix *dzDW = newMatrix(layer->outputDim, layer->inputDim);

  for (size_t r = 0; r < layer->outputDim; r++) {
    float g = getMatrixValue(loss, r, 0);
    for (size_t c = 0; c < layer->inputDim; c++) {
      float res = g * getMatrixValue(input, c, 0);

      if (res > MAXFLOAT) {
        res = MAXFLOAT;
      }

      setMatrixValue(dzDW, r, c, res);
    }
  }

  for (size_t c = 0; c < layer->inputDim; c++) {
    float total = 0.0;
    float in = getMatrixValue(input, c, 0);
    for (size_t r = 0; r < layer->outputDim; r++) {
      total += getMatrixValue(layer->weights, r, c) *
               getMatrixValue(loss, r, 0) * in;
    }

    if (total > MAXFLOAT) {
      total = MAXFLOAT;
    }

    setMatrixValue(layer->loss, c, 0, total);
  }

  matrixAdd(dzDW, layer->weightsGrad, layer->weightsGrad);
  matrixAdd(loss, layer->biasGrad, layer->biasGrad);
  deleteMatrix(dzDW);
}

void nnLinearLayerOptimize(NNLinearLayer *layer, float steps, float lr,
                           float decay) {
  float inv_steps = 1 / steps;
  float decay_scale = 1.0 - decay;

  Matrix *dB = newMatrix(layer->outputDim, 1);
  Matrix *dW = newMatrix(layer->outputDim, layer->inputDim);

  fillMatrix(dB, 0.0f);
  matrixScale(layer->biasGrad, inv_steps, dB);
  matrixAdd(layer->prevBiasGrad, dB, dB);
  matrixScale(dB, decay_scale, layer->prevBiasGrad);

  matrixScale(dB, -lr, dB);
  matrixAdd(layer->bias, dB, layer->bias);

  fillMatrix(dW, 0.0f);
  matrixScale(layer->weightsGrad, inv_steps, dW);
  matrixAdd(layer->prevWeightsGrad, dW, dW);
  matrixScale(dW, decay_scale, layer->prevWeightsGrad);

  matrixScale(dW, -lr, dW);
  matrixAdd(layer->weights, dW, layer->weights);

  fillMatrix(layer->weightsGrad, 0.0);
  fillMatrix(layer->biasGrad, 0.0);
  deleteMatrix(dB);
  deleteMatrix(dW);
}

NNLinearLayer *newLinearLayer(size_t inputDim, size_t outputDim) {
  NNLinearLayer *layer = malloc(sizeof(NNLinearLayer));

  if (layer == NULL)
    return NULL;

  *layer = (NNLinearLayer){
      .inputDim = inputDim,
      .outputDim = outputDim,
      .weights = newMatrix(outputDim, inputDim),
      .weightsGrad = newMatrix(outputDim, inputDim),
      .prevWeightsGrad = newMatrix(outputDim, inputDim),
      .bias = newMatrix(outputDim, 1),
      .biasGrad = newMatrix(outputDim, 1),
      .prevBiasGrad = newMatrix(outputDim, 1),
      .output = newMatrix(outputDim, 1),
      .loss = newMatrix(inputDim, 1),
      .forward = nnLinearLayerForward,
      .backward = nnLinearLayerBackward,
      .optimize = nnLinearLayerOptimize,
  };

  fillMatrix(layer->output, 0.0);
  fillMatrix(layer->weightsGrad, 0.0);
  fillMatrix(layer->biasGrad, 0.0);
  return layer;
}

void destroyLinearLayer(NNLinearLayer *layer) {
  deleteMatrix(layer->weights);
  deleteMatrix(layer->weightsGrad);
  deleteMatrix(layer->prevWeightsGrad);
  deleteMatrix(layer->bias);
  deleteMatrix(layer->biasGrad);
  deleteMatrix(layer->prevBiasGrad);
  deleteMatrix(layer->output);
  deleteMatrix(layer->loss);
  free(layer);
}

void initLinearLayer(NNLinearLayer *nnLayer, float min, float max) {
  randomMatrix(nnLayer->weights, min, max);
  randomMatrix(nnLayer->bias, min, max);
}
