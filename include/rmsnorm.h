#ifndef RMS_NORM_H
#define RMS_NORM_H
#include "activation.h"

void rmsNormForward(NNActivationLayer *layer, const Matrix *in);
void rmsNormBackward(NNActivationLayer *layer, const Matrix *in,
                     const Matrix *loss);
NNActivationLayer *newRMSNormLayer(Vec2 dim);
#endif
