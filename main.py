import pandas as pd
import rnn
import os
import numpy as np
import re

def load_dataset(parquet_path, vocab_size=5000):
    """
    Load and preprocess the dataset from a Parquet file.

    Parameters:
    - parquet_path: str, path to the Parquet file
    - vocab_size: int, maximum vocabulary size (default: 5000)

    Returns:
    - X_indices: np.ndarray, token indices for the training data
    - unique_tokens: list of str, unique tokens in the vocabulary
    - token_to_idx: dict, mapping from token to index
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found at {parquet_path}")

    # Load Parquet file
    df = pd.read_parquet(parquet_path)
    texts = df['text'].tolist()

    # Tokenize and add EOS tokens after sentences
    EOS_TOKEN = "<EOS>"
    tokens = []
    for text in texts:
        # Split text into sentences using basic punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        for sentence in sentences:
            if sentence:  # Skip empty sentences
                sentence_tokens = sentence.split()
                tokens.extend(sentence_tokens)
                tokens.append(EOS_TOKEN)  # Add EOS after each sentence

    # Build vocabulary including EOS token
    unique_tokens = sorted(set(tokens))[:vocab_size - 1]  # Reserve one slot for EOS
    if EOS_TOKEN not in unique_tokens:
        unique_tokens.append(EOS_TOKEN)  # Ensure EOS is included
    token_to_idx = {token: idx for idx, token in enumerate(unique_tokens)}

    # Convert tokens to indices
    X_indices = np.array([token_to_idx.get(token, 0) for token in tokens], dtype=np.int32)

    return X_indices, unique_tokens, token_to_idx

def main():
    """
    Train the RNN model using the WikiText dataset and perform inference on a test example.
    """
    # Get the directory of this script (RNN-school/)
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Define the path to the Parquet file
    data_dir = os.path.join(project_root, "WikiText_data", "WikiText-2-v1")
    parquet_path = os.path.join(data_dir, "train-00000-of-00001.parquet")

    # Load and preprocess the dataset
    print("Loading dataset...")
    X_indices, unique_tokens, token_to_idx = load_dataset(parquet_path)
    print("Dataset loaded.")

    # Define RNN hyperparameters
    embedding_dim = 50
    hidden_size = 100
    epochs = 100
    learning_rate = 0.0001
    max_input_size = 1000  # Limit for testing

    # Limit input size for testing
    if len(X_indices) > max_input_size:
        X_indices = X_indices[:max_input_size]

    # Prepare X as token indices cast to float32
    X = X_indices.astype(np.float32)

    # Initialize embeddings and weight matrices
    actual_vocab_size = len(unique_tokens)
    embeddings = np.random.randn(actual_vocab_size, embedding_dim).astype(np.float32) * 0.01
    Wxh = np.random.randn(embedding_dim, hidden_size).astype(np.float32) * 0.01
    Whh = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.01
    Why = np.random.randn(hidden_size, actual_vocab_size).astype(np.float32) * 0.01
    bh = np.zeros(hidden_size, dtype=np.float32)
    by = np.zeros(actual_vocab_size, dtype=np.float32)

    # Initialize the trainer with the DLL path
    dll_path = os.path.join(project_root, "src", "rnn.dll")
    trainer = rnn.RNNTrainer(dll_path=dll_path)
    trainer.set_vocabulary(unique_tokens)  # Set the vocabulary for inference

    # Train the RNN and save weights
    print("Training model...")
    trainer.train(
        X=X,
        embeddings=embeddings,
        Wxh=Wxh,
        Whh=Whh,
        Why=Why,
        bh=bh,
        by=by,
        epochs=epochs,
        learning_rate=learning_rate,
        save_weights=True
    )
    print("Training completed.")

    # Inference on test example
    test_input = "I am"
    print(f"Performing inference on: '{test_input}'")
    generated_text = trainer.inference(test_input)
    print("Generated text:", generated_text)

if __name__ == "__main__":
    main()