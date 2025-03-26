#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "matrix.h"
#include "backprop.h"

__declspec(dllexport) float train(params *p, float *embedded_X, int *targets, int sequence_length, int embedding_dim, int epochs, float learning_rate, float *d_embedded_X)
{
    // Allocate memory
    float *h = calloc(p->hidden_size * sequence_length, sizeof(float));
    float *h_prev = calloc(p->hidden_size, sizeof(float));
    float *output = malloc(p->vocab_size * sizeof(float));
    float *ys = malloc(p->vocab_size * sequence_length * sizeof(float));
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

    float total_loss = 0.0;
    printf("Starting training with %d epochs\n", epochs);
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        total_loss = 0.0;
        memset(h_prev, 0, p->hidden_size * sizeof(float));

        // Forward pass
        for (int t = 0; t < sequence_length; t++)
        {
            float *x = &embedded_X[t * embedding_dim];
            float *h_t = &h[t * p->hidden_size];
            float *y_t = &ys[t * p->vocab_size];

            mat_vec_mul(p->Wxh, x, h_t, p->hidden_size, embedding_dim);
            float *temp = malloc(p->hidden_size * sizeof(float));
            mat_vec_mul(p->Whh, h_prev, temp, p->hidden_size, p->hidden_size);
            vec_add_scaled(h_t, temp, p->hidden_size, 1.0);
            free(temp);
            vec_add_scaled(h_t, p->bh, p->hidden_size, 1.0);
            tanh_vec(h_t, p->hidden_size);

            mat_vec_mul(p->Why, h_t, output, p->vocab_size, p->hidden_size);
            vec_add_scaled(output, p->by, p->vocab_size, 1.0);
            softmax_vec(output, p->vocab_size);
            memcpy(y_t, output, p->vocab_size * sizeof(float));

            if (t < sequence_length - 1)
            {
                total_loss += cross_entropy(output, targets[t + 1], p->vocab_size);
            }

            memcpy(h_prev, h_t, p->hidden_size * sizeof(float));
        }

        backprop(p, embedded_X, targets, h, ys, sequence_length, dWxh, dWhh, dWhy, dbh, dby, d_embedded_X);

        vec_add_scaled(p->Wxh, dWxh, p->embedding_dim * p->hidden_size, -learning_rate);
        vec_add_scaled(p->Whh, dWhh, p->hidden_size * p->hidden_size, -learning_rate);
        vec_add_scaled(p->Why, dWhy, p->hidden_size * p->vocab_size, -learning_rate);
        vec_add_scaled(p->bh, dbh, p->hidden_size, -learning_rate);
        vec_add_scaled(p->by, dby, p->vocab_size, -learning_rate);

        printf("Epoch %d, Loss: %f\n", epoch, total_loss / (sequence_length - 1));
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

    return total_loss / (sequence_length - 1);
}

int sample_from_probs(float *probs, int size)
{
    float r = (float)rand() / RAND_MAX;
    float cumulative = 0.0;
    for (int i = 0; i < size; i++)
    {
        cumulative += probs[i];
        if (r <= cumulative)
            return i;
    }
    return size - 1;
}

__declspec(dllexport) float *generate(params *p, float *embeddings, int vocab_size, int start_idx, int eos_idx, int max_len, int *gen_len)
{
    float *h = calloc(p->hidden_size, sizeof(float));
    float *output = malloc(p->vocab_size * sizeof(float));
    float *generated = malloc(max_len * sizeof(float));
    int len = 0;
    int current_idx = start_idx;

    if (!h || !output || !generated)
    {
        fprintf(stderr, "Memory allocation failed in generate()\n");
        free(h);
        free(output);
        free(generated);
        exit(1);
    }

    while (len < max_len)
    {
        float *x = &embeddings[current_idx * p->embedding_dim];
        mat_vec_mul(p->Wxh, x, h, p->hidden_size, p->embedding_dim);
        float *temp = malloc(p->hidden_size * sizeof(float));
        if (!temp)
        {
            fprintf(stderr, "Temporary memory allocation failed in generate()\n");
            free(h);
            free(output);
            free(generated);
            exit(1);
        }
        mat_vec_mul(p->Whh, h, temp, p->hidden_size, p->hidden_size);
        vec_add_scaled(h, temp, p->hidden_size, 1.0);
        free(temp);
        vec_add_scaled(h, p->bh, p->hidden_size, 1.0);
        tanh_vec(h, p->hidden_size);

        mat_vec_mul(p->Why, h, output, p->vocab_size, p->hidden_size);
        vec_add_scaled(output, p->by, p->vocab_size, 1.0);
        softmax_vec(output, p->vocab_size);

        current_idx = sample_from_probs(output, p->vocab_size);
        generated[len++] = (float)current_idx;

        if (current_idx == eos_idx)
            break;
    }

    *gen_len = len;

    free(h);
    free(output);

    if (len == 0)
    {
        free(generated);
        return NULL;
    }

    float *result = realloc(generated, len * sizeof(float));
    if (!result)
    {
        fprintf(stderr, "Reallocation failed in generate()\n");
        free(generated);
        exit(1);
    }
    return result;
}

__declspec(dllexport) void free_generated(float *ptr)
{
    free(ptr);
}