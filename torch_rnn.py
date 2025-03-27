import pandas as pd
import os
import numpy as np
import re
import gensim.downloader as api
import torch
import torch.nn as nn
import torch.optim as optim

# Data loading and preprocessing functions (reused from the original code)
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

# Define the PyTorch RNN model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, embedding_matrix):
        super(RNNModel, self).__init__()
        # Embedding layer with pre-trained Word2Vec weights (frozen)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False  # Freeze embeddings
        # RNN layer
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        # Fully connected layer to map RNN outputs to vocabulary size
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        """
        Forward pass through the model.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            hidden: Initial hidden state (optional)
        Returns:
            output: Logits of shape (batch_size, sequence_length, vocab_size)
            hidden: Final hidden state
        """
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        output, hidden = self.rnn(x, hidden)  # output: (batch_size, seq_len, hidden_size)
        output = self.fc(output)  # (batch_size, seq_len, vocab_size)
        return output, hidden

# Inference function to generate text
def generate(model, start_text, token_to_idx, idx_to_token, max_length=50):
    """
    Generate text starting from an input string.
    """
    model.eval()
    # Convert start text to token indices
    start_tokens = [token_to_idx.get(token, 0) for token in start_text.split() if token in token_to_idx]
    if not start_tokens:
        start_tokens = [0]  # Default to index 0 if input is invalid
    tokens = start_tokens.copy()
    hidden = None
    
    with torch.no_grad():
        for _ in range(max_length):
            input_tensor = torch.tensor([tokens[-1]], dtype=torch.long).unsqueeze(0)  # (1, 1)
            output, hidden = model(input_tensor, hidden)  # output: (1, 1, vocab_size)
            _, predicted = torch.max(output, dim=2)  # (1, 1)
            next_token = predicted.item()
            tokens.append(next_token)
            if next_token == token_to_idx.get('<EOS>', -1):  # Stop at EOS
                break
    
    # Convert indices back to text
    generated_text = ' '.join([idx_to_token.get(idx, '<UNK>') for idx in tokens])
    return generated_text

def main():
    """
    Train the PyTorch RNN model with pre-trained Word2Vec embeddings and perform inference.
    """
    # Set up paths
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
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}
    print(f"Training dataset loaded with {vocab_size} unique tokens.")

    # Load validation and test data (optional, kept for completeness)
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
    embedding_dim = word2vec_model.vector_size  # 300
    print("Word2Vec model loaded.")

    # Create embedding matrix
    print("Creating embedding matrix...")
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
    for token, idx in token_to_idx.items():
        if token in word2vec_model:
            embedding_matrix[idx] = word2vec_model[token]
        else:
            embedding_matrix[idx] = np.random.randn(embedding_dim).astype(np.float32) * 0.01
    print("Embedding matrix created.")

    # Tokenize training data and limit size
    X_indices = tokenize_data(training_texts, token_to_idx)
    print(f"Training dataset tokenized with {len(X_indices)} tokens.")
    max_input_size = 5000
    if len(X_indices) > max_input_size:
        X_indices = X_indices[:max_input_size + 1]

    # Prepare input and target sequences
    X_indices_input = X_indices[:-1]
    targets = X_indices[1:]
    input_seq = torch.tensor(X_indices_input, dtype=torch.long)  # (seq_len)
    target_seq = torch.tensor(targets, dtype=torch.long)  # (seq_len)

    # Define hyperparameters
    hidden_size = 100
    epochs = 3000
    learning_rate = 0.00001

    # Initialize the model
    model = RNNModel(vocab_size, embedding_dim, hidden_size, embedding_matrix)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print("Training model...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs, _ = model(input_seq.unsqueeze(0))  # (1, seq_len, vocab_size)
        outputs = outputs.squeeze(0)  # (seq_len, vocab_size)
        loss = criterion(outputs.view(-1, vocab_size), target_seq.view(-1))
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    print(f"Training completed. Final Loss: {loss.item():.4f}")

    # Inference
    test_input = "I am"
    print(f"Performing inference on: '{test_input}'")
    generated_text = generate(model, test_input, token_to_idx, idx_to_token)
    print("Generated text:", generated_text)

    # Save the trained model
    weights_file = "trained_model_with_word2vec.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'unique_tokens': unique_tokens,
        'token_to_idx': token_to_idx
    }, weights_file)
    print(f"Model weights and vocabulary saved to '{weights_file}'.")

if __name__ == "__main__":
    main()