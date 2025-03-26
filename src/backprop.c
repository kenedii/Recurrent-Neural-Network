#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "matrix.h"
#include "params.h"
#include "backprop.h"

void backprop(params *p, float *embedded_X, int *targets, float *h, float *ys, int sequence_length,
              float *dWxh, float *dWhh, float *dWhy, float *dbh, float *dby, float *d_embedded_X)
{
    float *dh_next = calloc(p->hidden_size, sizeof(float));
    float *dy = malloc(p->vocab_size * sizeof(float));
    float *dh = malloc(p->hidden_size * sizeof(float));

    if (!dh_next || !dy || !dh)
    {
        fprintf(stderr, "Memory allocation failed in backprop\n");
        exit(1);
    }

    memset(dWxh, 0, p->embedding_dim * p->hidden_size * sizeof(float));
    memset(dWhh, 0, p->hidden_size * p->hidden_size * sizeof(float));
    memset(dWhy, 0, p->hidden_size * p->vocab_size * sizeof(float));
    memset(dbh, 0, p->hidden_size * sizeof(float));
    memset(dby, 0, p->vocab_size * sizeof(float));
    memset(d_embedded_X, 0, sequence_length * p->embedding_dim * sizeof(float));

    for (int t = sequence_length - 1; t >= 0; t--)
    {
        float *x = &embedded_X[t * p->embedding_dim];
        float *h_t = &h[t * p->hidden_size];
        float *h_prev = (t > 0) ? &h[(t - 1) * p->hidden_size] : calloc(p->hidden_size, sizeof(float));
        float *y_t = &ys[t * p->vocab_size];

        if (t < sequence_length - 1)
        {
            memcpy(dy, y_t, p->vocab_size * sizeof(float));
            dy[targets[t + 1]] -= 1.0;
        }
        else
        {
            memset(dy, 0, p->vocab_size * sizeof(float));
        }

        add_outer_product(dWhy, dy, h_t, p->vocab_size, p->hidden_size, 1.0);
        vec_add_scaled(dby, dy, p->vocab_size, 1.0);

        trans_mat_vec_mul(p->Why, dy, dh, p->vocab_size, p->hidden_size);

        for (int i = 0; i < p->hidden_size; i++)
        {
            float tanh_deriv = 1.0 - h_t[i] * h_t[i];
            dh[i] = (dh[i] + dh_next[i]) * tanh_deriv;
        }

        add_outer_product(dWxh, x, dh, p->embedding_dim, p->hidden_size, 1.0);
        add_outer_product(dWhh, h_prev, dh, p->hidden_size, p->hidden_size, 1.0);
        vec_add_scaled(dbh, dh, p->hidden_size, 1.0);

        trans_mat_vec_mul(p->Wxh, dh, &d_embedded_X[t * p->embedding_dim], p->embedding_dim, p->hidden_size);

        trans_mat_vec_mul(p->Whh, dh, dh_next, p->hidden_size, p->hidden_size);

        if (t == 0)
            free(h_prev);
    }

    free(dh_next);
    free(dy);
    free(dh);
}