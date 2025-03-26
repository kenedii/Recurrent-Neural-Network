import ctypes
import numpy as np
from time import time

# Define the Params structure to match the C struct (without embeddings)
class Params(ctypes.Structure):
    _fields_ = [
        ("vocab_size", ctypes.c_int),
        ("embedding_dim", ctypes.c_int),
        ("hidden_size", ctypes.c_int),
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
        
        # Set up the train function
        self.rnn_lib.train.argtypes = [
            ctypes.POINTER(Params),           # params
            ctypes.POINTER(ctypes.c_float),   # embedded_X
            ctypes.POINTER(ctypes.c_int),     # targets
            ctypes.c_int,                     # sequence_length
            ctypes.c_int,                     # embedding_dim
            ctypes.c_int,                     # epochs
            ctypes.c_float,                   # learning_rate
            ctypes.POINTER(ctypes.c_float)    # d_embedded_X
        ]
        self.rnn_lib.train.restype = ctypes.c_float
        
        # Set up generate function
        self.rnn_lib.generate.argtypes = [
            ctypes.POINTER(Params),           # params
            ctypes.POINTER(ctypes.c_float),   # embeddings
            ctypes.c_int,                     # vocab_size
            ctypes.c_int,                     # start_idx
            ctypes.c_int,                     # eos_idx
            ctypes.c_int,                     # max_len
            ctypes.POINTER(ctypes.c_int),     # gen_len
        ]
        self.rnn_lib.generate.restype = ctypes.POINTER(ctypes.c_float)
        
        # Set up free_generated function
        self.rnn_lib.free_generated.argtypes = [ctypes.POINTER(ctypes.c_float)]
        self.rnn_lib.free_generated.restype = None

    def set_vocabulary(self, unique_tokens):
        """
        Set the vocabulary for token-to-index and index-to-token mappings.

        Parameters:
        - unique_tokens: list of str, unique tokens in the vocabulary
        """
        self.unique_tokens = unique_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(unique_tokens)}
        self.idx_to_token = {idx: token for idx, token in enumerate(unique_tokens)}
        self.vocab_size = len(unique_tokens)

    def train(self, embedded_X, targets, sequence_length, embedding_dim, epochs, learning_rate, Wxh=None, Whh=None, Why=None, bh=None, by=None, save_weights=True):
        """
        Train the RNN using the DLL and save weights to a file after training.

        Parameters:
        - embedded_X: np.ndarray, embedded input vectors [sequence_length, embedding_dim], float32
        - targets: np.ndarray, target token indices [sequence_length], int32
        - sequence_length: int, length of the input sequence
        - embedding_dim: int, dimension of each embedding vector
        - epochs: int, number of training epochs (this call will run all epochs in the C function)
        - learning_rate: float, learning rate for training
        - Wxh, Whh, Why, bh, by: np.ndarray, optional initial weights and biases
        - save_weights: bool, whether to save weights after training

        Returns:
        - float: average loss
        - np.ndarray: gradients for embedded_X
        """
        # Ensure arrays are contiguous and of correct type
        embedded_X = np.ascontiguousarray(embedded_X, dtype=np.float32)
        targets = np.ascontiguousarray(targets, dtype=np.int32)

        # Initialize weights if not provided
        if Wxh is None:
            Wxh = np.random.randn(embedding_dim, self.hidden_size).astype(np.float32) * 0.01
        if Whh is None:
            Whh = np.random.randn(self.hidden_size, self.hidden_size).astype(np.float32) * 0.01
        if Why is None:
            Why = np.random.randn(self.hidden_size, self.vocab_size).astype(np.float32) * 0.01
        if bh is None:
            bh = np.zeros(self.hidden_size, dtype=np.float32)
        if by is None:
            by = np.zeros(self.vocab_size, dtype=np.float32)

        # Ensure weights are contiguous
        Wxh = np.ascontiguousarray(Wxh, dtype=np.float32)
        Whh = np.ascontiguousarray(Whh, dtype=np.float32)
        Why = np.ascontiguousarray(Why, dtype=np.float32)
        bh = np.ascontiguousarray(bh, dtype=np.float32)
        by = np.ascontiguousarray(by, dtype=np.float32)

        # Validate dimensions
        if embedded_X.shape[0] != sequence_length or embedded_X.shape[1] != embedding_dim:
            raise ValueError(f"embedded_X shape {embedded_X.shape} does not match sequence_length {sequence_length} and embedding_dim {embedding_dim}")
        if targets.shape[0] != sequence_length:
            raise ValueError(f"targets shape {targets.shape} does not match sequence_length {sequence_length}")
        if Wxh.shape != (embedding_dim, self.hidden_size) or Whh.shape != (self.hidden_size, self.hidden_size) or Why.shape != (self.hidden_size, self.vocab_size):
            raise ValueError("Weight matrix dimensions do not match expected shapes")
        if bh.shape[0] != self.hidden_size or by.shape[0] != self.vocab_size:
            raise ValueError("Bias vector dimensions do not match expected sizes")

        # Create and populate the Params structure
        p = Params()
        p.vocab_size = self.vocab_size
        p.embedding_dim = embedding_dim
        p.hidden_size = self.hidden_size
        p.Wxh = Wxh.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p.Whh = Whh.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p.Why = Why.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p.bh = bh.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p.by = by.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Allocate gradient array
        d_embedded_X = np.zeros_like(embedded_X, dtype=np.float32)

        # Print training setup
        print("Training setup:")
        print(f" - Vocab size: {self.vocab_size}")
        print(f" - Embedding dim: {embedding_dim}")
        print(f" - Hidden size: {self.hidden_size}")
        print(f" - Sequence length: {sequence_length}")
        print(f" - Epochs: {epochs}")
        print(f" - Learning rate: {learning_rate}")

        # Call the train function from the DLL
        print("Starting training...")
        loss = self.rnn_lib.train(
            ctypes.byref(p),
            embedded_X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            targets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            sequence_length,
            embedding_dim,
            epochs,  # Here we pass the full number of epochs
            learning_rate,
            d_embedded_X.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        print(f"Training completed successfully. Loss: {loss:.4f}")

        # Store weights in the instance
        self.Wxh = Wxh
        self.Whh = Whh
        self.Why = Why
        self.bh = bh
        self.by = by

        if save_weights:
            weights_file = f"rnn_weights_{int(time())}.npz"
            print(f"Saving weights to '{weights_file}'...")
            try:
                np.savez(
                    weights_file,
                    Wxh=Wxh,
                    Whh=Whh,
                    Why=Why,
                    bh=bh,
                    by=by,
                    unique_tokens=np.array(self.unique_tokens, dtype=object)
                )
                print(f"Weights saved successfully to '{weights_file}'.")
            except Exception as e:
                print(f"Error saving weights: {e}")

        return loss, d_embedded_X

    def load_model(self, weights_file):
        """
        Load model weights, vocabulary, and embeddings from a file.

        Parameters:
        - weights_file: str, path to the .npz file containing the weights and embeddings
        """
        data = np.load(weights_file, allow_pickle=True)
        self.Wxh = data['Wxh']
        self.Whh = data['Whh']
        self.Why = data['Why']
        self.bh = data['bh']
        self.by = data['by']
        self.hidden_size = self.Whh.shape[0]
        self.vocab_size = self.by.shape[0]
        self.embedding_dim = self.Wxh.shape[0]
        if 'unique_tokens' in data:
            self.unique_tokens = data['unique_tokens'].tolist()
            self.token_to_idx = {token: idx for idx, token in enumerate(self.unique_tokens)}
            self.idx_to_token = {idx: token for idx, token in enumerate(self.unique_tokens)}
        else:
            raise ValueError("No vocabulary found in the weights file.")
        if 'embeddings' in data:
            self.embeddings = data['embeddings']
        else:
            raise ValueError("No embeddings found in the weights file.")

    def inference(self, input_text, max_len=50):
        """
        Perform inference on text input using the DLL with loaded embeddings.

        Parameters:
        - input_text: str, input text to condition the generation
        - max_len: int, maximum length of the generated sequence

        Returns:
        - str, generated text
        """
        if not hasattr(self, 'unique_tokens'):
            raise ValueError("Vocabulary not set. Call set_vocabulary or load_model with vocabulary first.")
        if not hasattr(self, 'Wxh'):
            raise ValueError("Model weights not loaded. Call load_model or train first.")
        if not hasattr(self, 'embeddings'):
            raise ValueError("Embeddings not loaded. Ensure the model was loaded with embeddings.")

        # Tokenize the input text
        input_tokens = input_text.split()
        if not input_tokens:
            raise ValueError("Input text is empty.")
        start_idx = self.token_to_idx.get(input_tokens[-1], 0)
        eos_idx = self.token_to_idx["<EOS>"]

        # Use the loaded embeddings
        embeddings = self.embeddings

        # Ensure embeddings are contiguous and correct type
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.shape[0] != self.vocab_size or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embeddings shape {embeddings.shape} does not match vocab_size {self.vocab_size} and embedding_dim {self.embedding_dim}")

        # Prepare the Params structure
        p = Params()
        p.vocab_size = self.vocab_size
        p.embedding_dim = self.embedding_dim
        p.hidden_size = self.hidden_size
        p.Wxh = self.Wxh.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p.Whh = self.Whh.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p.Why = self.Why.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p.bh = self.bh.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p.by = self.by.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Call the generate function
        gen_len = ctypes.c_int()
        generated_ptr = self.rnn_lib.generate(
            ctypes.byref(p),
            embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.vocab_size,
            start_idx,
            eos_idx,
            max_len,
            ctypes.byref(gen_len)
        )

        if not generated_ptr:
            print("Warning: generate returned NULL, no sequence generated.")
            return ""

        # Convert the generated sequence to a NumPy array and free memory
        generated = np.ctypeslib.as_array(generated_ptr, shape=(gen_len.value,))
        generated = generated.astype(int)  # Convert to integer indices
        result = generated.copy()
        self.rnn_lib.free_generated(generated_ptr)

        # Convert indices to tokens
        generated_tokens = [self.idx_to_token.get(idx, "<UNK>") for idx in result]
        return ' '.join(generated_tokens)

    def set_model_params(self, vocab_size, embedding_dim, hidden_size):
        """
        Set the model parameters for initialization.

        Parameters:
        - vocab_size: int, size of the vocabulary
        - embedding_dim: int, dimension of embedding vectors
        - hidden_size: int, size of the hidden state
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
