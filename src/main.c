#include "linear.h"
#include "matrix.h"
#include "rmsnorm.h"
#include "softmax.h"
#include <math.h>
#include <stdint.h>
#include <sys/syslimits.h>
#include <time.h>
#define EPS 1e-8
#define BATCHSIZE 20

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
  deleteLinearLayer(linearLayer1);
  deleteLinearLayer(linearLayer2);
  deleteLinearLayer(linearLayer3);
  deleteActivationLayer(activLayer1);
  deleteActivationLayer(activLayer2);
  deleteActivationLayer(activLayer3);
  return 0;
}

void initIDXFile(const char *path, FILE **f, Vec4 *dim) {
  if ((*f = fopen(path, "rb")) == NULL) {
    fprintf(stderr, "[ERROR] Error opening file %s\n", path);
    exit(1);
  }

  uint8_t magicNumber[4];
  fread(magicNumber, sizeof(int), 1, *f);
  int numDims = magicNumber[3];

  for (int i = 0; i < 4; i++) {
    if (i < numDims) {
      uint8_t data[4];
      fread(data, sizeof(int), 1, *f);
      ((size_t *)(dim))[i] = byteswap(data);
    } else {
      ((size_t *)(dim))[i] = 1;
    }
  }
}

size_t nextIDXValue(FILE *f, Vec4 dim, uint8_t *out) {
  size_t size = dim.x * dim.y * dim.z * dim.w;
  size_t read = fread(out, sizeof(uint8_t), size, f);
  return size == read;
}

bool loadNextImage(FILE *images, FILE *labels, Matrix *img, Matrix *label) {
  uint8_t buffer[BUFSIZ];
  if (!nextIDXValue(images, img->dim, buffer)) {
    fprintf(stderr, "[ERROR] failed to load image\n");
    return false;
  }

  initMatrix_uint8(img, buffer);

  if (!nextIDXValue(labels, (Vec4){1, 1, 1, 1}, buffer)) {
    fprintf(stderr, "[ERROR] failed to load label\n");
    return false;
  }

  fillMatrix(label, 0.0f);
  setMatrixValue(label, (Vec4){0, 0, 0, buffer[0]}, 1.0f);
  return true;
}

int nnMNISTImageClassifier() {
  FILE *images, *labels;
  Vec4 imgDim;
  Vec4 lblDim;

  initIDXFile("./TrainingData/train-images-idx3-ubyte.bin", &images, &imgDim);
  initIDXFile("./TrainingData/train-labels-idx1-ubyte.bin", &labels, &lblDim);

  Matrix *img = newMatrix((Vec4){1, 1, imgDim.y, imgDim.z});
  Matrix *target = newMatrix((Vec4){1, 1, 1, 10});
  Matrix *loss = newMatrix((Vec4){1, 1, 1, 10});

  NNLinearLayer *layer1 = newLinearLayer((Vec2){28, 28}, (Vec2){1, 32});
  initLinearLayer(layer1, -1.0f, 1.0f);
  NNActivationLayer *rms1 = newRMSNormLayer((Vec2){1, 32});

  NNLinearLayer *layer2 = newLinearLayer((Vec2){1, 32}, (Vec2){1, 32});
  initLinearLayer(layer2, -1.0f, 1.0f);
  NNActivationLayer *rms2 = newRMSNormLayer((Vec2){1, 32});

  NNLinearLayer *layer3 = newLinearLayer((Vec2){1, 32}, (Vec2){1, 10});
  initLinearLayer(layer3, -1.0f, 1.0f);
  NNActivationLayer *softmax = newSoftmaxActivationLayer((Vec2){1, 10});

  int outerLoop = imgDim.x / BATCHSIZE;

  for (int i = 0; i < outerLoop; i++) {
    float totalLoss = 0.0f;

    for (int j = 0; j < BATCHSIZE; j++) {
      if (!loadNextImage(images, labels, img, target)) {
        return 1;
      }
      matrixScale(img, 1.0f / 255.0f, img);

      layer1->forward(layer1, img);
      rms1->forward(rms1, layer1->output);
      layer2->forward(layer2, rms1->output);
      rms2->forward(rms2, layer2->output);
      layer3->forward(layer3, rms2->output);
      softmax->forward(softmax, layer3->output);

      totalLoss += crossEntropy(target, softmax->output);

      crossEntropyDeriv(target, softmax->output, loss);
      layer3->backward(layer3, rms2->output, loss);
      rms2->backward(rms2, layer2->output, layer3->loss);
      layer2->backward(layer2, rms1->output, rms2->loss);
      rms1->backward(rms1, layer1->output, layer2->loss);
      layer1->backward(layer1, img, rms1->loss);
    }

    float avgLoss = totalLoss / BATCHSIZE;

    if ((i + 1) % 100 == 0) {
      printf("%d/%d\nLoss: %f\n", (i + 1), outerLoop, avgLoss);
      displayMatrix("Prediction", softmax->output);
    }

    if (avgLoss < 0.01f) {
      printf("\rLoss: %f\n", avgLoss);
      break;
    }

    layer1->optimize(layer1, BATCHSIZE, 0.1f, 0.1f);
    layer2->optimize(layer2, BATCHSIZE, 0.1f, 0.1f);
    layer3->optimize(layer2, BATCHSIZE, 0.1f, 0.1f);
  }

  fclose(images);
  fclose(labels);

  return 0;

  initIDXFile("./TrainingData/t10k-images-idx3-ubyte.bin", &images, &imgDim);
  initIDXFile("./TrainingData/t10k-labels-idx1-ubyte.bin", &labels, &lblDim);

  float totalLoss = 0.0f;
  for (int i = 0; i < imgDim.x; i++) {
    loadNextImage(images, labels, img, target);
    matrixScale(img, 1.0f / 255.0f, img);

    layer1->forward(layer1, img);
    rms1->forward(rms1, layer1->output);
    layer2->forward(layer2, rms1->output);
    rms2->forward(rms2, layer2->output);
    layer3->forward(layer3, rms2->output);
    softmax->forward(softmax, layer3->output);

    float loss = crossEntropy(target, softmax->output);
    totalLoss += loss;

    if (i % 500 == 0) {
      printf("Loss: %.2f\n", loss);
      displayMatrixf("Target", "%3.2f ", target);
      displayMatrixf("Label", "%3.2f ", softmax->output);
      displayMatrixf("Image", "%3.0f", img);
    }
  }
  printf("\rAvg Loss: %.2f\n", totalLoss / imgDim.x);

  deleteMatrix(img);
  deleteMatrix(target);
  deleteMatrix(loss);
  deleteLinearLayer(layer1);
  deleteActivationLayer(rms1);
  deleteLinearLayer(layer2);
  deleteActivationLayer(rms2);
  deleteLinearLayer(layer3);
  deleteActivationLayer(softmax);
  fclose(images);
  fclose(labels);
  return 0;
}

int main(void) { return nnMNISTImageClassifier(); }
