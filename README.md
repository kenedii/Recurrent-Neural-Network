# Recurrent Neural Network (RNN) Implementation

An RNN implemented in C with Python FastAPI endpoints.

[Trained on WikiText-103-v1 and WikiText-2-v1](https://huggingface.co/datasets/Salesforce/wikitext/)

---

## Installation

1. **Install required packages**:

    ```bash
    pip install -r requirements.txt
    ```

---

## Operating Systems

### Windows
**C RNN**: Ensure that the "dll path" in `main.py` points to `lib/rnnlib.dll`

#### Training:
- **C RNN**:

    ```bash
    python main.py
    ```

- **PyTorch**:

    ```bash
    python torch_rnn.py
    ```

#### Inference:
- Run the FastAPI server with the following command:

    ```bash
    uvicorn api:app --reload --port=<port_number>
    ```

---

### Docker / Linux

#### Training:
- **C RNN**: Ensure that the "dll path" in `main.py` points to `lib/rnnlib.so`

    ```bash
    python main.py
    ```

- **PyTorch**:

    ```bash
    python torch_rnn.py
    ```

#### Inference:
- Ensure that the "dll path" in `api.py` points to `lib/rnnlib.so`

    ```bash
    uvicorn api:app --reload --port=<port_number>
    ```

---

## Backend / RNN Implementations

1. **PyTorch**:
    - The `torch_rnn.py` file creates an RNN class that extends `torch.nn.Module` to train an RNN model and generate text.
    - Backpropagation is handled automatically using PyTorch's automatic differentiation engine (`torch.autograd`).

2. **C Backend**:
    - The `rnn.py` file uses a compiled C library (`rnn.dll` or `rnnlib.so`) to train the RNN model and perform inference (generate text).
    - Weights are stored to compute backpropagation through time.

---

## Loss Function

- **Cross-Entropy Loss** (Negative Log Likelihood)

---

## FastAPI Endpoints

- **Generate using C RNN library**
- **Generate using PyTorch**
- **Upload new weights file** to the server’s weights folder.

---

## Deploying

- **Docker image** created with `Dockerfile`
- **Deployed to Google Cloud Run** using the `gcloud` CLI
- **Requirements**:
  - At least **1 GB memory** and **1 CPU core**.

---

## Training

- **C Code**: Trained on WikiText-103-v1
- **PyTorch Code**: Trained on WikiText-2-v1
- **Tokenization**: Only word (letters only) tokens are kept. `<EOS>` token is added after every sentence.
- **Embedding**: Tokens are embedded using pretrained `word2vec-google-news-300`.

---

### Hyperparameters

- **learning_rate** = `0.00001`: Low learning rate to control gradients. LR >= 0.0001 leads to "inf" loss (exploding gradient).
- **epochs** = `600`: Trained for high epochs since weight update steps (learning rate) are very small.
- **hidden_size** = `100`: Refers to the number of neurons in the hidden state of the RNN. 
    - Input weights are of size `(embed_dim, hidden_size)`, and hidden state weights are of size `(hidden_size, hidden_size)`.

- **Model weights and word embeddings** are saved to the weights file after training.

---

## Inference

### C Implementation:

- **Cumulative Distribution Function (CDF) sampling**:
    1. Generates a random number `r`.
    2. Iterates through probabilities and adds `probs[i]` to the cumulative sum.
    3. Returns the smallest index `i` where the cumulative probability is greater than or equal to `r`.

### PyTorch Implementation:

- **Greedy Sampling**:
    - Takes the token with the highest probability from the softmax output: 

    ```python
    torch.max(output)
    ```

---
