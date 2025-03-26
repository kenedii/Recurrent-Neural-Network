#ifndef BACKPROP_H
#define BACKPROP_H

#include "params.h" // Assuming this defines the params struct

void backprop(params *p, float *embedded_X, int *targets, float *h, float *ys, int sequence_length,
              float *dWxh, float *dWhh, float *dWhy, float *dbh, float *dby, float *d_embedded_X);

#endif // BACKPROP_H