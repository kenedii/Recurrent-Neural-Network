#ifndef BACKPROP_H
#define BACKPROP_H

typedef struct params params;

void backprop(params *p, float *X, float *h, float *ys, int input_size,
              float *dWxh, float *dWhh, float *dWhy, float *dbh, float *dby);

#endif