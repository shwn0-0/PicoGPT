#include "softmax.h"
#include <math.h>
#define EPS 1e-8

void softmax(NNActivationLayer *layer, const Matrix *in) {
  if (matrixSize(in) != matrixSize(layer->output)) {
    printf("%lu, %lu, %lu, %lu", in->dim.x, in->dim.y, in->dim.z, in->dim.w);
    fprintf(stderr, "[ERROR] dimension mismatch when performing softmax\n");
    exit(1);
  }

  size_t size = matrixSize(in);
  float total = 0.0;

  for (size_t i = 0; i < size; i++) {
    float e = exp(in->data[i]);

    if (e == INFINITY)
      e = MAXFLOAT;

    layer->output->data[i] = e;
    total += e;
  }

  float total_inv = 1 / (total + EPS);

  for (size_t i = 0; i < size; i++) {
    layer->output->data[i] *= total_inv;
  }
}

void softmaxDeriv(NNActivationLayer *layer, const Matrix *in,
                  const Matrix *loss) {
  if (matrixSize(in) != matrixSize(layer->loss)) {
    printf("%lu, %lu, %lu, %lu", in->dim.x, in->dim.y, in->dim.z, in->dim.w);
    fprintf(stderr, "[ERROR] dimension mismatch when performing softmax\n");
    exit(1);
  }

  size_t size = matrixSize(in);
  for (size_t i = 0; i < size; i++) {
    layer->loss->data[i] = 0.0f;

    for (size_t j = 0; j < size; j++) {
      layer->loss->data[i] +=
          layer->output->data[i] * (i == j - layer->output->data[j]);
    }

    layer->loss->data[i] *= loss->data[i];
  }
}

NNActivationLayer *newSoftmaxActivationLayer(Vec2 inputDim) {
  Vec4 dim = {1, 1, inputDim.x, inputDim.y};
  NNActivationLayer *layer = malloc(sizeof(NNActivationLayer));
  *layer = (NNActivationLayer){
      .forward = softmax,
      .backward = softmaxDeriv,
      .output = newMatrix(dim),
      .loss = newMatrix(dim),
  };
  return layer;
}

void deleteActivationLayer(NNActivationLayer *layer) {
  deleteMatrix(layer->output);
  deleteMatrix(layer->loss);
  free(layer);
}
