#include "softmax.h"
#include <math.h>
#define EPS 1e-8

void softmax(NNActivationLayer *layer, const Matrix *in) {
  if (matrixSize(in) != matrixSize(layer->output)) {
    fprintf(stderr, "[ERROR] dimension mismatch when performing softmax\n");
    exit(1);
  }

  Matrix *total = newMatrix((Vec4){1, in->dim.x, in->dim.y, in->dim.z});
  Matrix *max = newMatrix((Vec4){1, in->dim.x, in->dim.y, in->dim.z});

  for (size_t a = 0; a < in->dim.x; a++)
    for (size_t b = 0; b < in->dim.y; b++)
      for (size_t i = 0; i < in->dim.z; i++) {
        float rowMax = 0.0f;

        for (size_t j = 0; j < in->dim.w; j++) {
          float val = getMatrixValue(in, (Vec4){a, b, i, j});

          if (val > rowMax) {
            rowMax = val;
          }
        }

        setMatrixValue(max, (Vec4){0, a, b, i}, rowMax);
      }

  for (size_t a = 0; a < in->dim.x; a++)
    for (size_t b = 0; b < in->dim.y; b++)
      for (size_t i = 0; i < in->dim.z; i++) {
        float rowTotal = 0.0f;
        float rowMax = getMatrixValue(max, (Vec4){0, a, b, i});

        for (size_t j = 0; j < in->dim.w; j++) {
          float val = getMatrixValue(in, (Vec4){a, b, i, j});
          float e = exp(val - rowMax);

          if (e >= INFINITY)
            e = MAXFLOAT;

          setMatrixValue(layer->output, (Vec4){a, b, i, j}, e);
          rowTotal += e;
        }

        setMatrixValue(total, (Vec4){0, a, b, i}, rowTotal);
      }

  for (size_t a = 0; a < in->dim.x; a++)
    for (size_t b = 0; b < in->dim.y; b++)
      for (size_t i = 0; i < in->dim.z; i++) {
        float rowTotal = getMatrixValue(total, (Vec4){0, a, b, i});
        float totalInv = 1 / (rowTotal + EPS);

        for (size_t j = 0; j < in->dim.w; j++) {
          float val = getMatrixValue(layer->output, (Vec4){a, b, i, j});
          setMatrixValue(layer->output, (Vec4){a, b, i, j}, val * totalInv);
        }
      }

  deleteMatrix(total);
  deleteMatrix(max);
}

void softmaxDeriv(NNActivationLayer *layer, const Matrix *in,
                  const Matrix *loss) {
  if (matrixSize(in) != matrixSize(layer->loss)) {
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
