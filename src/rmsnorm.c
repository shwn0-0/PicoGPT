#include "rmsnorm.h"
#include "activation.h"
#include <math.h>
#define EPS 1e-8

typedef struct NNRMSActivationLayer {
  NNActivationLayer activation;
  float rms;
  // TODO: explore affine parameters
} NNRMSActivationLayer;

void rmsNormForward(NNActivationLayer *layer, const Matrix *in) {
  if (matrixSize(in) != matrixSize(layer->output)) {
    fprintf(stderr, "[ERROR] dimension mismatch when performing rms norm\n");
    exit(1);
  }

  NNRMSActivationLayer *rmsLayer = (NNRMSActivationLayer *)layer;

  size_t size = matrixSize(in);
  float meanSqr = 0.0f;
  float sizeInv = 1.0 / (float)size;

  for (size_t i = 0; i < size; i++) {
    meanSqr += (in->data[i] * in->data[i]) * sizeInv;
  }

  rmsLayer->rms = sqrtf(meanSqr + EPS);
  float rmsInv = 1.0f / rmsLayer->rms;
  for (size_t i = 0; i < size; i++) {
    layer->output->data[i] = in->data[i] * rmsInv;
  }
}
void rmsNormBackward(NNActivationLayer *layer, const Matrix *in,
                     const Matrix *loss) {
  if (matrixSize(layer->loss) != matrixSize(loss)) {
    fprintf(stderr,
            "[ERROR] dimension mismatch when performing rms norm backprop\n");
    exit(1);
  }

  NNRMSActivationLayer *rmsLayer = (NNRMSActivationLayer *)layer;

  size_t size = matrixSize(loss);
  float rmsInv = 1.0f / rmsLayer->rms;
  float sizeInv = 1.0f / (float)size;

  float c = 0.0f;
  for (int i = 0; i < size; i++) {
    c += (loss->data[i] * layer->output->data[i]) * sizeInv;
  }

  for (int i = 0; i < size; i++) {
    layer->loss->data[i] =
        rmsInv * (loss->data[i] - c * layer->output->data[i]);
  }
}

NNActivationLayer *newRMSNormLayer(Vec2 dim) {
  NNRMSActivationLayer *layer = malloc(sizeof(NNRMSActivationLayer));
  if (layer == NULL)
    return NULL;

  *layer = (NNRMSActivationLayer){
      .activation =
          {
              .forward = rmsNormForward,
              .backward = rmsNormBackward,
              .output = newMatrix((Vec4){1, 1, dim.x, dim.y}),
              .loss = newMatrix((Vec4){1, 1, dim.x, dim.y}),
          },
      .rms = 0.0f,
  };
  return (NNActivationLayer *)layer;
}
