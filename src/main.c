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

float crossEntropy(const Matrix *target, const Matrix *prediction) {
  if (!target || !prediction || matrixSize(target) != matrixSize(prediction))
    return 0.0;

  size_t size = matrixSize(target);
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
  if (!target || !prediction || matrixSize(target) != matrixSize(prediction) ||
      matrixSize(prediction) != matrixSize(out))
    return false;

  size_t size = matrixSize(target);

  for (size_t i = 0; i < size; i++) {
    float y = target->data[i];
    float y_hat = prediction->data[i];
    // out->data[i] = -(y / (y_hat + EPS));
    // TODO: This is the derivative assuming softmax
    out->data[i] = y_hat - y;
  }

  return true;
}

int nnDemultiplexer() {
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

  Matrix *input = newMatrix((Vec4){.x = 1, .y = 1, .z = 1, .w = 3});
  Matrix *target = newMatrix((Vec4){.x = 1, .y = 1, .z = 1, .w = 8});
  Matrix *loss = newMatrix((Vec4){.x = 1, .y = 1, .z = 1, .w = 8});

  if (!input || !target || !loss) {
    return 2;
  }

  NNLinearLayer *linearLayer1 = newLinearLayer((Vec2){1, 3}, (Vec2){1, 8});
  initLinearLayer(linearLayer1, -0.5, 0.5);
  NNActivationLayer *activLayer1 = newSoftmaxActivationLayer((Vec2){1, 8});

  NNLinearLayer *linearLayer2 = newLinearLayer((Vec2){1, 8}, (Vec2){1, 8});
  initLinearLayer(linearLayer2, -0.5, 0.5);
  NNActivationLayer *activLayer2 = newSoftmaxActivationLayer((Vec2){1, 8});

  NNLinearLayer *linearLayer3 = newLinearLayer((Vec2){1, 8}, (Vec2){1, 8});
  initLinearLayer(linearLayer3, -0.5, -0.5);
  NNActivationLayer *activLayer3 = newSoftmaxActivationLayer((Vec2){1, 8});

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

    if (avg_loss < 0.001) {
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
  // destroyLinearLayer(linearLayer2);
  // destroyLinearLayer(linearLayer3);
  deleteActivationLayer(activLayer1);
  // deleteActivationLayer(activLayer2);
  // deleteActivationLayer(activLayer3);
  return 0;
}

void initIDXFile(const char *path, FILE **f, Vec4 *dim) {
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

size_t nextIDXValue(FILE *f, Vec4 dim, uint8_t *out) {
  size_t size = dim.y * dim.z * dim.w;
  size_t read = fread(out, sizeof(uint8_t), size, f);
  return size == read;
}

int nnMNISTImageClassifier() {
  FILE *images, *labels;
  Vec4 imgDim;
  Vec4 lblDim;

  initIDXFile("./TrainingData/train-images-idx3-ubyte.bin", &images, &imgDim);
  initIDXFile("./TrainingData/train-labels-idx1-ubyte.bin", &labels, &lblDim);

  printf("Image Dim: %lu x %lu x %lu\n", imgDim.x, imgDim.y, imgDim.z);
  printf("Label Dim: %lu\n\n", lblDim.x);

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

int main(void) {
  nnDemultiplexer();
  return 0;
}
