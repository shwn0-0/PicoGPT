#include "linear.h"
#include <math.h>

void nnLinearLayerForward(NNLinearLayer *layer, Matrix *input) {
  if (!matrixMul(input, layer->weights, layer->output)) {
    exit(1);
  }

  if (!matrixAdd(layer->bias, layer->output, layer->output)) {
    exit(1);
  }
}

void nnLinearLayerBackward(NNLinearLayer *layer, Matrix *input, Matrix *loss) {
  Matrix *dzDW = newMatrix(layer->weightsGrad->dim);

  for (int i = 0; i < dzDW->dim.z; i++)
    for (size_t j = 0; j < dzDW->dim.w; j++) {
      float g = getMatrixValue(loss, (Vec4){0, 0, i, j});

      for (size_t r = 0; r < dzDW->dim.x; r++)
        for (size_t c = 0; c < dzDW->dim.y; c++) {
          float res = g * getMatrixValue(input, (Vec4){0, 0, c, r});

          if (res > MAXFLOAT) {
            res = MAXFLOAT;
          }

          setMatrixValue(dzDW, (Vec4){r, c, i, j}, res);
        }
    }

  for (size_t r = 0; r < layer->weights->dim.x; r++)
    for (size_t c = 0; c < layer->weights->dim.y; c++) {
      float total = 0.0;
      float in = getMatrixValue(input, (Vec4){0, 0, c, r});

      for (size_t i = 0; i < layer->weights->dim.z; i++) {
        for (size_t j = 0; j < layer->weights->dim.w; j++) {
          total += getMatrixValue(layer->weights, (Vec4){r, c, i, j}) *
                   getMatrixValue(loss, (Vec4){0, 0, i, j}) * in;
        }
      }
      setMatrixValue(layer->loss, (Vec4){0, 0, c, r}, total);
    }

  matrixAdd(dzDW, layer->weightsGrad, layer->weightsGrad);
  matrixAdd(loss, layer->biasGrad, layer->biasGrad);
  deleteMatrix(dzDW);
}

void nnLinearLayerOptimize(NNLinearLayer *layer, float steps, float lr,
                           float decay) {
  float inv_steps = 1 / steps;
  float decay_scale = 1.0 - decay;

  Matrix *dB = newMatrix(layer->bias->dim);
  Matrix *dW = newMatrix(layer->weights->dim);

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

NNLinearLayer *newLinearLayer(Vec2 inputDim, Vec2 outputDim) {
  NNLinearLayer *layer = malloc(sizeof(NNLinearLayer));

  if (layer == NULL)
    return NULL;

  Vec4 weightsDim = {inputDim.y, inputDim.x, outputDim.x, outputDim.y};
  Vec4 biasDim = {1, 1, outputDim.x, outputDim.y};
  Vec4 lossDim = {1, 1, inputDim.x, inputDim.y};

  *layer = (NNLinearLayer){
      .weights = newMatrix(weightsDim),
      .weightsGrad = newMatrix(weightsDim),
      .prevWeightsGrad = newMatrix(weightsDim),
      .bias = newMatrix(biasDim),
      .biasGrad = newMatrix(biasDim),
      .prevBiasGrad = newMatrix(biasDim),
      .output = newMatrix(biasDim),
      .loss = newMatrix(lossDim),
      .forward = nnLinearLayerForward,
      .backward = nnLinearLayerBackward,
      .optimize = nnLinearLayerOptimize,
  };

  fillMatrix(layer->output, 0.0);
  fillMatrix(layer->weightsGrad, 0.0);
  fillMatrix(layer->biasGrad, 0.0);
  return layer;
}

void deleteLinearLayer(NNLinearLayer *layer) {
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
