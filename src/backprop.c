#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h> // for memset and memcpy
#include "matrix.h"

typedef struct params
{
    int vocab_size;
    int embedding_dim;
    int hidden_size;
    float *embeddings;
    float *Wxh;
    float *Whh;
    float *Why;
    float *bh;
    float *by;
} params;

void backprop(params *p, float *X, float *h, float *ys, int input_size,
              float *dWxh, float *dWhh, float *dWhy, float *dbh, float *dby)
{
    float *dh_next = calloc(p->hidden_size, sizeof(float));
    float *dy = malloc(p->vocab_size * sizeof(float));
    float *dh = malloc(p->hidden_size * sizeof(float));

    if (!dh_next || !dy || !dh)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // Clear gradients
    memset(dWxh, 0, p->embedding_dim * p->hidden_size * sizeof(float));
    memset(dWhh, 0, p->hidden_size * p->hidden_size * sizeof(float));
    memset(dWhy, 0, p->hidden_size * p->vocab_size * sizeof(float));
    memset(dbh, 0, p->hidden_size * sizeof(float));
    memset(dby, 0, p->vocab_size * sizeof(float));

    // Backward pass through time
    for (int t = input_size - 1; t >= 0; t--)
    {
        float *x = &p->embeddings[(int)X[t] * p->embedding_dim];
        float *h_t = &h[t * p->hidden_size];
        float *h_prev = (t > 0) ? &h[(t - 1) * p->hidden_size] : calloc(p->hidden_size, sizeof(float));
        float *y_t = &ys[t * p->vocab_size];

        // Compute output gradient
        if (t < input_size - 1)
        {
            memcpy(dy, y_t, p->vocab_size * sizeof(float));
            dy[(int)X[t + 1]] -= 1.0; // dy = y_t - one_hot(target)
        }
        else
        {
            memset(dy, 0, p->vocab_size * sizeof(float)); // No gradient at last step
        }

        // Gradient w.r.t Why and by
        add_outer_product(dWhy, dy, h_t, p->vocab_size, p->hidden_size, 1.0);
        vec_add_scaled(dby, dy, p->vocab_size, 1.0);

        // Gradient w.r.t hidden state
        trans_mat_vec_mul(p->Why, dy, dh, p->vocab_size, p->hidden_size);

        // Add contribution from next time step and apply tanh derivative
        for (int i = 0; i < p->hidden_size; i++)
        {
            float tanh_deriv = 1.0 - h_t[i] * h_t[i];
            dh[i] = (dh[i] + dh_next[i]) * tanh_deriv;
        }

        // Gradient w.r.t Wxh, Whh, bh
        add_outer_product(dWxh, x, dh, p->embedding_dim, p->hidden_size, 1.0);
        add_outer_product(dWhh, h_prev, dh, p->hidden_size, p->hidden_size, 1.0);
        vec_add_scaled(dbh, dh, p->hidden_size, 1.0);

        // Compute dh_next
        trans_mat_vec_mul(p->Whh, dh, dh_next, p->hidden_size, p->hidden_size);

        if (t == 0)
            free(h_prev);
    }

    free(dh_next);
    free(dy);
    free(dh);
}