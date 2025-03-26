#ifndef PARAMS_H
#define PARAMS_H

typedef struct params
{
    int vocab_size;    // Number of unique tokens
    int embedding_dim; // Size of embedding vectors
    int hidden_size;   // Size of hidden state
    float *Wxh;        // Input-to-hidden weights [embedding_dim x hidden_size]
    float *Whh;        // Hidden-to-hidden weights [hidden_size x hidden_size]
    float *Why;        // Hidden-to-output weights [hidden_size x vocab_size]
    float *bh;         // Hidden bias [hidden_size]
    float *by;         // Output bias [vocab_size]
} params;

#endif // PARAMS_H