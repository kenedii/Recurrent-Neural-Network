#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "matrix.h" // Include custom matrix operations
#include "backprop.h"

typedef struct params
{
    int vocab_size;    // Number of unique tokens
    int embedding_dim; // Size of embedding vectors
    int hidden_size;   // Size of hidden state
    float *embeddings; // [vocab_size x embedding_dim]
    float *Wxh;        // [embedding_dim x hidden_size]
    float *Whh;        // [hidden_size x hidden_size]
    float *Why;        // [hidden_size x vocab_size]
    float *bh;         // [hidden_size]
    float *by;         // [vocab_size]
} params;

float tanh_activation(float x)
{
    return (expf(x) - expf(-x)) / (expf(x) + expf(-x));
}

void train(params *p, float *X, int input_size, int epochs, float learning_rate)
{
    // Allocate memory for hidden states, outputs, and output probabilities
    float *h = calloc(p->hidden_size * input_size, sizeof(float));
    float *h_prev = calloc(p->hidden_size, sizeof(float));
    float *output = malloc(p->vocab_size * sizeof(float));
    float *ys = malloc(p->vocab_size * input_size * sizeof(float)); // Store outputs for backprop

    // Gradient accumulators
    float *dWxh = calloc(p->embedding_dim * p->hidden_size, sizeof(float));
    float *dWhh = calloc(p->hidden_size * p->hidden_size, sizeof(float));
    float *dWhy = calloc(p->hidden_size * p->vocab_size, sizeof(float));
    float *dbh = calloc(p->hidden_size, sizeof(float));
    float *dby = calloc(p->vocab_size, sizeof(float));

    if (!h || !h_prev || !output || !ys || !dWxh || !dWhh || !dWhy || !dbh || !dby)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float total_loss = 0.0;
        memset(h_prev, 0, p->hidden_size * sizeof(float));

        // Forward pass
        for (int t = 0; t < input_size; t++)
        {
            // Get input embedding
            float *x = &p->embeddings[(int)X[t] * p->embedding_dim];
            float *h_t = &h[t * p->hidden_size];
            float *y_t = &ys[t * p->vocab_size];

            // h_t = tanh(Wxh * x + Whh * h_prev + bh)
            mat_vec_mul(p->Wxh, x, h_t, p->hidden_size, p->embedding_dim);
            float *temp = malloc(p->hidden_size * sizeof(float));
            mat_vec_mul(p->Whh, h_prev, temp, p->hidden_size, p->hidden_size);
            vec_add_scaled(h_t, temp, p->hidden_size, 1.0);
            free(temp);
            vec_add_scaled(h_t, p->bh, p->hidden_size, 1.0);
            tanh_vec(h_t, p->hidden_size);

            // Output: y_t = softmax(Why * h_t + by)
            mat_vec_mul(p->Why, h_t, output, p->vocab_size, p->hidden_size);
            vec_add_scaled(output, p->by, p->vocab_size, 1.0);
            softmax_vec(output, p->vocab_size);
            memcpy(y_t, output, p->vocab_size * sizeof(float));

            // Calculate loss (next word prediction)
            if (t < input_size - 1)
            {
                total_loss += cross_entropy(output, (int)X[t + 1], p->vocab_size);
            }

            // Update h_prev
            memcpy(h_prev, h_t, p->hidden_size * sizeof(float));
        }

        // Backpropagation
        backprop(p, X, h, ys, input_size, dWxh, dWhh, dWhy, dbh, dby);

        // Update parameters
        vec_add_scaled(p->Wxh, dWxh, p->embedding_dim * p->hidden_size, -learning_rate);
        vec_add_scaled(p->Whh, dWhh, p->hidden_size * p->hidden_size, -learning_rate);
        vec_add_scaled(p->Why, dWhy, p->hidden_size * p->vocab_size, -learning_rate);
        vec_add_scaled(p->bh, dbh, p->hidden_size, -learning_rate);
        vec_add_scaled(p->by, dby, p->vocab_size, -learning_rate);

        printf("Epoch %d, Loss: %f\n", epoch, total_loss / (input_size - 1));
    }

    free(h);
    free(h_prev);
    free(output);
    free(ys);
    free(dWxh);
    free(dWhh);
    free(dWhy);
    free(dbh);
    free(dby);
}