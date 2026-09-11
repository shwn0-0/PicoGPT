#ifndef SOFTMAX_H
#define SOFTMAX_H
#include "activation.h"

NNActivationLayer * newSoftmaxActivationLayer(size_t inputDim,
                                             size_t embeddingDim);
#endif
