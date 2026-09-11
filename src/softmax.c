#include "softmax.h"
#define EPS 1e-8

void softmax(NNActivationLayer *layer, const Matrix *in) {
  size_t size = in->rows * in->cols;
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
  size_t size = in->rows * in->cols;
  for (size_t i = 0; i < size; i++) {
    layer->loss->data[i] = 0.0f;

    for (size_t j = 0; j < size; j++) {
      layer->loss->data[i] +=
          layer->output->data[i] * (i == j - layer->output->data[j]);
    }

    layer->loss->data[i] *= loss->data[i];
  }
}

NNActivationLayer *newSoftmaxActivationLayer(size_t dim) {
  NNActivationLayer *layer = malloc(sizeof(NNActivationLayer));
  *layer = (NNActivationLayer){
      .forward = softmax,
      .backward = softmaxDeriv,
      .output = newMatrix(dim, 1),
      .loss = newMatrix(dim, 1),
  };
  return layer;
}

void deleteActivationLayer(NNActivationLayer *layer) {
  deleteMatrix(layer->output);
  deleteMatrix(layer->loss);
  free(layer);
}
