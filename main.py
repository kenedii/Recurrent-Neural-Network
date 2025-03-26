import pandas as pd
import os
import numpy as np
import re
import gensim.downloader as api
from rnn import RNNTrainer  # Your updated module
from time import time

def load_data(parquet_paths):
    """
    Load text data from one or more Parquet files.
    """
    if isinstance(parquet_paths, str):
        paths = [parquet_paths]
    else:
        paths = list(parquet_paths)
    all_texts = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Parquet file not found at {path}")
        df = pd.read_parquet(path)
        all_texts.extend(df['text'].to_list())
    return all_texts

def build_vocabulary(texts, vocab_size=5000):
    """
    Build vocabulary from text data, including EOS token.
    """
    EOS_TOKEN = "<EOS>"
    tokens = []
    for text in texts:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        for sentence in sentences:
            if sentence:
                sentence_tokens = sentence.split()
                tokens.extend(sentence_tokens)
                tokens.append(EOS_TOKEN)
    unique_tokens = sorted(set(tokens))[:vocab_size - 1]
    if EOS_TOKEN not in unique_tokens:
        unique_tokens.append(EOS_TOKEN)
    token_to_idx = {token: idx for idx, token in enumerate(unique_tokens)}
    return unique_tokens, token_to_idx

def tokenize_data(texts, token_to_idx):
    """
    Convert text data to token indices using the provided vocabulary.
    """
    EOS_TOKEN = "<EOS>"
    tokens = []
    for text in texts:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        for sentence in sentences:
            if sentence:
                sentence_tokens = sentence.split()
                tokens.extend(sentence_tokens)
                tokens.append(EOS_TOKEN)
    X_indices = np.array([token_to_idx.get(token, 0) for token in tokens], dtype=np.int32)
    return X_indices

def main():
    """
    Train the RNN model with pre-trained Word2Vec embeddings and perform inference.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, "WikiText_data", "WikiText-103-v1")
    train_paths = [
        os.path.join(data_dir, "train-00000-of-00002.parquet"),
        os.path.join(data_dir, "train-00001-of-00002.parquet")
    ]

    # Load and preprocess training data
    print("Loading training datasets...")
    training_texts = load_data(train_paths)
    unique_tokens, token_to_idx = build_vocabulary(training_texts)
    vocab_size = len(unique_tokens)
    print(f"Training dataset loaded with {len(unique_tokens)} unique tokens.")

    # Load validation and test data
    validation_path = os.path.join(data_dir, "validation-00000-of-00001.parquet")
    test_path = os.path.join(data_dir, "test-00000-of-00001.parquet")
    
    print("Loading validation dataset...")
    validation_texts = load_data(validation_path)
    validation_X_indices = tokenize_data(validation_texts, token_to_idx)
    print(f"Validation dataset loaded with {len(validation_X_indices)} tokens.")

    print("Loading test dataset...")
    test_texts = load_data(test_path)
    test_X_indices = tokenize_data(test_texts, token_to_idx)
    print(f"Test dataset loaded with {len(test_X_indices)} tokens.")

    # Load pre-trained Word2Vec model
    print("Loading pre-trained Word2Vec model...")
    word2vec_model = api.load("word2vec-google-news-300")
    print("Word2Vec model loaded.")

    # Set embedding dimension from Word2Vec model
    embedding_dim = word2vec_model.vector_size  # 300 for Google News model

    # Create fixed embedding matrix
    print("Creating embedding matrix...")
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
    for token, idx in token_to_idx.items():
        if token in word2vec_model:
            embedding_matrix[idx] = word2vec_model[token]
        else:
            embedding_matrix[idx] = np.random.randn(embedding_dim).astype(np.float32) * 0.01
    print("Embedding matrix created.")

    # Tokenize training data and prepare input
    X_indices = tokenize_data(training_texts, token_to_idx)
    print(f"Training dataset tokenized with {len(X_indices)} tokens.")

    # Limit training input size
    max_input_size = 5000
    if len(X_indices) > max_input_size:
        X_indices = X_indices[:max_input_size + 1]

    # Prepare input and target sequences
    X_indices_input = X_indices[:-1]
    targets = X_indices[1:]
    targets_np = targets.astype(np.int32)

    # Convert indices to fixed embeddings
    embedded_X_np = embedding_matrix[X_indices_input]

    # Define hyperparameters
    hidden_size = 100
    epochs = 200
    learning_rate = 0.00001

    # Initialize RNN weights with updated embedding_dim
    Wxh = np.random.randn(embedding_dim, hidden_size).astype(np.float32) * 0.01
    Whh = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.01
    Why = np.random.randn(hidden_size, vocab_size).astype(np.float32) * 0.01
    bh = np.zeros(hidden_size, dtype=np.float32)
    by = np.zeros(vocab_size, dtype=np.float32)

    # Initialize the trainer
    dll_path = "rnn.dll"
    trainer = RNNTrainer(dll_path=dll_path)
    trainer.set_vocabulary(unique_tokens)
    trainer.set_model_params(vocab_size, embedding_dim, hidden_size)

    # Train the model
    print("Training model...")
    loss, _ = trainer.train(
        embedded_X_np,
        targets_np,
        len(X_indices_input),
        embedding_dim,
        epochs,
        learning_rate,
        Wxh=Wxh,
        Whh=Whh,
        Why=Why,
        bh=bh,
        by=by
    )
    print(f"Training completed. Final Loss: {loss:.4f}")

    # Inference
    test_input = "I am"
    print(f"Performing inference on: '{test_input}'")
    generated_text = trainer.inference(test_input, embedding_matrix)
    print("Generated text:", generated_text)

    # Save trained model with fixed embeddings
    weights_file = "trained_model_with_word2vec.npz"
    np.savez(
        weights_file,
        embeddings=embedding_matrix,
        Wxh=Wxh,
        Whh=Whh,
        Why=Why,
        bh=bh,
        by=by,
        unique_tokens=np.array(unique_tokens, dtype=object)
    )
    print(f"Model weights and embeddings saved to '{weights_file}'.")

if __name__ == "__main__":
    main()
