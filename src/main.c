#include "linear.h"
#include "matrix.h"
#include "softmax.h"
#include <math.h>
#include <stdint.h>
#include <sys/syslimits.h>
#include <time.h>
#define EPS 1e-8

#define byteswap(data)                                                         \
  (((uint32_t)data[0] << 24) | ((uint32_t)data[1] << 16) |                     \
   ((uint32_t)data[2] << 8) | ((uint32_t)data[3] << 0))

float sigmoid_func(float x) {
  if (x >= 0.0)
    return 1.0 / (1.0 + exp(-x));

  float e = exp(x);
  return e / (1.0 + e);
}

bool sigmoid(const Matrix *m, Matrix *out) {
  if (!m || !out || m->rows != out->rows || m->cols != out->cols)
    return false;

  size_t size = m->rows * m->cols;
  for (size_t i = 0; i < size; i++) {
    out->data[i] = sigmoid_func(m->data[i]);
  }

  return true;
}

bool sigmoidDeriv(const Matrix *m, Matrix *out) {
  size_t size = m->rows * m->cols;
  for (size_t i = 0; i < size; i++) {
    float s = m->data[i];
    out->data[i] = s * (1.0 - s);
  }
  return true;
}

// float swish(float x) {
//   return x * sigmoid_func(x);
// }

// float swishDeriv(float x) {
//   float s = sigmoid_func(x);
//   return s * (1.0 + x * (1.0 - s));
// }

float crossEntropy(const Matrix *target, const Matrix *prediction) {
  if (!target || !prediction || target->rows != prediction->rows ||
      target->cols != prediction->cols)
    return 0.0;

  size_t size = target->rows * target->cols;
  float loss = 0.0;

  for (size_t i = 0; i < size; i++) {
    float y = target->data[i];
    float y_hat = prediction->data[i];
    loss += -y * log(y_hat + EPS);
  }

  return loss;
}

bool crossEntropyDeriv(const Matrix *target, const Matrix *prediction,
                       Matrix *out) {
  if (!target || !prediction || target->rows != prediction->rows ||
      target->cols != prediction->cols || target->rows != out->rows ||
      target->cols != out->cols)
    return false;

  size_t size = target->rows * target->cols;

  for (size_t i = 0; i < size; i++) {
    float y = target->data[i];
    float y_hat = prediction->data[i];
    // out->data[i] = -(y / (y_hat + EPS));
    out->data[i] = y_hat - y; // TODO: This is the derivative assuming softmax
  }

  return true;
}

int demultiplexerNeuralNetwork() {
  float sampleData[][3] = {
      {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
      {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1},
  };

  float sampleTarget[][8] = {
      {1, 0, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0, 0},
      {0, 0, 1, 0, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0, 0},
      {0, 0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 0, 1, 0, 0},
      {0, 0, 0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 0, 0, 1},
  };

  srand(time(NULL));

  Matrix *input = newMatrix(3, 1);
  Matrix *target = newMatrix(8, 1);
  Matrix *loss = newMatrix(8, 1);

  NNLinearLayer *linearLayer1 = newLinearLayer(3, 8);
  initLinearLayer(linearLayer1, -1.0, 1.0);
  NNActivationLayer *activLayer1 = newSoftmaxActivationLayer(8);

  NNLinearLayer *linearLayer2 = newLinearLayer(8, 8);
  initLinearLayer(linearLayer2, -1.0, 1.0);
  NNActivationLayer *activLayer2 = newSoftmaxActivationLayer(8);

  NNLinearLayer *linearLayer3 = newLinearLayer(8, 8);
  initLinearLayer(linearLayer3, -1.0, 1.0);
  NNActivationLayer *activLayer3 = newSoftmaxActivationLayer(8);

  for (size_t i = 0; i < 100000; i++) {
    float lossTotal = 0.0;

    for (size_t k = 0; k < 8; k++) {
      initMatrix(input, sampleData[k]);
      initMatrix(target, sampleTarget[k]);

      linearLayer1->forward(linearLayer1, input);
      activLayer1->forward(activLayer1, linearLayer1->output);
      linearLayer2->forward(linearLayer2, activLayer1->output);
      activLayer2->forward(activLayer2, linearLayer2->output);
      linearLayer3->forward(linearLayer3, activLayer2->output);
      activLayer3->forward(activLayer3, linearLayer3->output);

      lossTotal += crossEntropy(target, activLayer3->output);

      crossEntropyDeriv(target, activLayer3->output, loss);

      // activLayer3->backward(activLayer3, linearLayer3->output, loss);
      linearLayer3->backward(linearLayer3, activLayer2->output, loss);
      activLayer2->backward(activLayer2, linearLayer2->output,
                            linearLayer3->loss);
      linearLayer2->backward(linearLayer2, activLayer1->output,
                             activLayer2->loss);
      activLayer1->backward(activLayer1, linearLayer1->output,
                            linearLayer2->loss);
      linearLayer1->backward(linearLayer1, input, activLayer1->loss);
    }

    float avg_loss = lossTotal / 8.0f;

    if (avg_loss < 0.0001) {
      printf("Loss: %f\n", avg_loss);
      break;
    }

    if (i % 1000 == 0) {
      printf("Loss: %f\n", avg_loss);
    }

    linearLayer3->optimize(linearLayer3, 8.0f, 0.1f, 0.0f);
    linearLayer2->optimize(linearLayer2, 8.0f, 0.1f, 0.0f);
    linearLayer1->optimize(linearLayer1, 8.0f, 0.1f, 0.0f);
  }
  putchar('\n');

  for (size_t i = 0; i < 8; i++) {
    initMatrix(input, sampleData[i]);
    initMatrix(target, sampleTarget[i]);

    linearLayer1->forward(linearLayer1, input);
    activLayer1->forward(activLayer1, linearLayer1->output);
    linearLayer2->forward(linearLayer2, activLayer1->output);
    activLayer2->forward(activLayer2, linearLayer2->output);
    linearLayer3->forward(linearLayer3, activLayer2->output);
    activLayer3->forward(activLayer3, linearLayer3->output);

    float loss_val = crossEntropy(target, activLayer3->output);

    printf("Test Run #%lu\n", i + 1);
    displayMatrix("Input", input);
    displayMatrix("Output", activLayer3->output);
    printf("Loss: %f\n\n", loss_val);
  }

  displayMatrix("W1", linearLayer1->weights);
  displayMatrix("W2", linearLayer2->weights);
  displayMatrix("W3", linearLayer3->weights);

  deleteMatrix(input);
  deleteMatrix(target);
  deleteMatrix(loss);
  destroyLinearLayer(linearLayer1);
  destroyLinearLayer(linearLayer2);
  destroyLinearLayer(linearLayer3);
  deleteActivationLayer(activLayer1);
  deleteActivationLayer(activLayer2);
  deleteActivationLayer(activLayer3);
  return 0;
}

typedef struct Dimension {
  int x;
  int y;
  int z;
  int w;
} Dimension;

void initIDXFile(const char *path, FILE **f, Dimension *dim) {
  if ((*f = fopen(path, "rb")) == NULL) {
    fprintf(stderr, "[FATAL] Error opening file %s\n", path);
    exit(1);
  }

  uint8_t magicNumber[4];
  fread(magicNumber, sizeof(int), 1, *f);
  int numDims = magicNumber[3];

  for (int i = 0; i < 4; i++) {
    if (i < numDims) {
      uint8_t data[4];
      fread(data, sizeof(int), 1, *f);
      ((int *)(dim))[i] = byteswap(data);
    } else {
      ((int *)(dim))[i] = 1;
    }
  }
}

size_t nextIDXValue(FILE *f, Dimension dim, uint8_t *out) {
  size_t size = dim.y * dim.z * dim.w;
  size_t read = fread(out, sizeof(uint8_t), size, f);
  return size == read;
}

int main(void) {
  FILE *images, *labels;
  Dimension imgDim;
  Dimension lblDim;

  initIDXFile("./TrainingData/train-images-idx3-ubyte.bin", &images, &imgDim);
  initIDXFile("./TrainingData/train-labels-idx1-ubyte.bin", &labels, &lblDim);

  printf("Image Dim: %d x %d x %d\n", imgDim.x, imgDim.y, imgDim.z);
  printf("Label Dim: %d\n\n", lblDim.x);

  for (int k = 0; k < 5; k++) {
    uint8_t imgData[28][28];
    uint8_t label;
    nextIDXValue(images, imgDim, (uint8_t *)imgData);
    nextIDXValue(labels, lblDim, &label);

    printf("\nSample Image: %d\n", label);
    for (int i = 0; i < 28; i++) {
      for (int j = 0; j < 28; j++) {
        printf("%02x ", imgData[i][j]);
      }
      putchar('\n');
    }
  }

  fclose(images);
  fclose(labels);
  return 0;
}
