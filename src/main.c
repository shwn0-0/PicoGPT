#include "linear.h"
#include "softmax.h"
#include <math.h>
#include <time.h>

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

int main(void) {
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
    float loss_total = 0.0;

    for (size_t k = 0; k < 8; k++) {
      initMatrix(input, sampleData[k]);
      initMatrix(target, sampleTarget[k]);

      linearLayer1->forward(linearLayer1, input);
      activLayer1->forward(activLayer1, linearLayer1->output);
      linearLayer2->forward(linearLayer2, activLayer1->output);
      activLayer2->forward(activLayer2, linearLayer2->output);
      linearLayer3->forward(linearLayer3, activLayer2->output);
      activLayer3->forward(activLayer3, linearLayer3->output);

      loss_total += crossEntropy(target, activLayer3->output);

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

    float avg_loss = loss_total / 8.0f;

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
