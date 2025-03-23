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

// Export the train function for Python access
__declspec(dllexport) void train(params *p, float *X, int input_size, int epochs, float learning_rate)
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

    // Check for memory allocation failures
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

    // Free allocated memory
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

// Helper function to sample from probability distribution (internal, not exported)
int sample_from_probs(float *probs, int size)
{
    float r = (float)rand() / RAND_MAX; // Random number between 0 and 1
    float cumulative = 0.0;
    for (int i = 0; i < size; i++)
    {
        cumulative += probs[i];
        if (r <= cumulative)
        {
            return i;
        }
    }
    return size - 1; // Default to last index if no selection (e.g., EOS)
}

// Export the generate function for Python access
__declspec(dllexport) float *generate(params *p, int start_idx, int eos_idx, int max_len, int *gen_len)
{
    /*
    Generate a sequence of tokens starting from start_idx until eos_idx or max_len is reached.
    - p: Pointer to trained model parameters
    - start_idx: Initial token index to start generation
    - eos_idx: Index of the EOS token to stop generation
    - max_len: Maximum length of the generated sequence
    - gen_len: Pointer to store the actual length of the generated sequence
    Returns: Dynamically allocated array of generated token indices
    */
    float *h = calloc(p->hidden_size, sizeof(float)); // Initial hidden state
    float *output = malloc(p->vocab_size * sizeof(float));
    float *generated = malloc(max_len * sizeof(float)); // Store generated indices
    int len = 0;
    int current_idx = start_idx;

    // Check for memory allocation failures
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
        // Get embedding for current token
        float *x = &p->embeddings[current_idx * p->embedding_dim];

        // Compute hidden state: h = tanh(Wxh * x + Whh * h + bh)
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

        // Compute output: softmax(Why * h + by)
        mat_vec_mul(p->Why, h, output, p->vocab_size, p->hidden_size);
        vec_add_scaled(output, p->by, p->vocab_size, 1.0);
        softmax_vec(output, p->vocab_size);

        // Sample the next token
        current_idx = sample_from_probs(output, p->vocab_size);
        generated[len++] = (float)current_idx;

        // Stop if EOS token is generated
        if (current_idx == eos_idx)
        {
            break;
        }
    }

    // Set the generated length
    *gen_len = len;

    // Clean up temporary allocations
    free(h);
    free(output);

    // Handle empty sequence case
    if (len == 0)
    {
        free(generated);
        return NULL; // Return NULL for empty sequence; Python should handle this
    }

    // Reallocate to exact size and return
    float *result = realloc(generated, len * sizeof(float));
    if (!result)
    {
        fprintf(stderr, "Reallocation failed in generate()\n");
        free(generated);
        exit(1);
    }
    return result;
}

// Export a function to free the generated memory for Python access
__declspec(dllexport) void free_generated(float *ptr)
{
    free(ptr);
}