import ctypes
import numpy as np
from time import time

# Define the Params structure to match the C struct
class Params(ctypes.Structure):
    _fields_ = [
        ("vocab_size", ctypes.c_int),
        ("embedding_dim", ctypes.c_int),
        ("hidden_size", ctypes.c_int),
        ("embeddings", ctypes.POINTER(ctypes.c_float)),
        ("Wxh", ctypes.POINTER(ctypes.c_float)),
        ("Whh", ctypes.POINTER(ctypes.c_float)),
        ("Why", ctypes.POINTER(ctypes.c_float)),
        ("bh", ctypes.POINTER(ctypes.c_float)),
        ("by", ctypes.POINTER(ctypes.c_float)),
    ]

class RNNTrainer:
    def __init__(self, dll_path):
        """Initialize the RNNTrainer by loading the DLL."""
        self.rnn_lib = ctypes.CDLL(dll_path)
        self.rnn_lib.train.argtypes = [
            ctypes.POINTER(Params),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]
        self.rnn_lib.train.restype = None

    def train(self, X, embeddings, Wxh, Whh, Why, bh, by, epochs, learning_rate,save_weights=True):
        """
        Train the RNN using the DLL and save weights to a file after training.

        Parameters:
        - X: np.ndarray, token indices as float32 [sequence_length]
        - embeddings: np.ndarray, [vocab_size, embedding_dim]
        - Wxh: np.ndarray, [embedding_dim, hidden_size]
        - Whh: np.ndarray, [hidden_size, hidden_size]
        - Why: np.ndarray, [hidden_size, vocab_size]
        - bh: np.ndarray, [hidden_size]
        - by: np.ndarray, [vocab_size]
        - epochs: int, number of training epochs
        - learning_rate: float, learning rate for training
        - weights_file: str, path to save the weights (default: 'rnn_weights.npz')
        """
        # Ensure arrays are contiguous and of type float32
        X = np.ascontiguousarray(X, dtype=np.float32)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        Wxh = np.ascontiguousarray(Wxh, dtype=np.float32)
        Whh = np.ascontiguousarray(Whh, dtype=np.float32)
        Why = np.ascontiguousarray(Why, dtype=np.float32)
        bh = np.ascontiguousarray(bh, dtype=np.float32)
        by = np.ascontiguousarray(by, dtype=np.float32)

        # Set dimensions
        input_size = len(X)
        vocab_size = embeddings.shape[0]
        embedding_dim = embeddings.shape[1]
        hidden_size = Wxh.shape[1]

        # Create and populate the Params structure
        p = Params()
        p.vocab_size = vocab_size
        p.embedding_dim = embedding_dim
        p.hidden_size = hidden_size
        p.embeddings = embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p.Wxh = Wxh.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p.Whh = Whh.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p.Why = Why.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p.bh = bh.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p.by = by.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Print training setup
        print("Training setup:")
        print(f" - Vocab size: {vocab_size}")
        print(f" - Embedding dim: {embedding_dim}")
        print(f" - Hidden size: {hidden_size}")
        print(f" - Input size: {input_size}")
        print(f" - Epochs: {epochs}")
        print(f" - Learning rate: {learning_rate}")
        print(f" - Sample of X (first 10): {X[:10]}")

        # Call the train function from the DLL
        print("Starting training...")
        self.rnn_lib.train(
            ctypes.byref(p),
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            input_size,
            epochs,
            learning_rate
        )
        print("Training completed successfully.")
        
        if save_weights:
            # Save weights to a file
            print(f"Saving weights to 'rnn_weights.npz'...")
            try:
                np.savez(
                    f"rnn_weights{str(time())}.npz",
                    embeddings=embeddings,
                    Wxh=Wxh,
                    Whh=Whh,
                    Why=Why,
                    bh=bh,
                    by=by
                )
                print(f"Weights saved successfully to 'rnn_weights.npz'.")
            except Exception as e:
                print(f"Error saving weights: {e}")
