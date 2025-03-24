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
        
        # Set up the train function
        self.rnn_lib.train.argtypes = [
            ctypes.POINTER(Params),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]
        self.rnn_lib.train.restype = None
        
        # Set up generate function
        self.rnn_lib.generate.argtypes = [
            ctypes.POINTER(Params),
            ctypes.c_int,                   # start_idx
            ctypes.c_int,                   # eos_idx
            ctypes.c_int,                   # max_len
            ctypes.POINTER(ctypes.c_int),   # gen_len
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

    def train(self, X, unique_tokens, embeddings, Wxh, Whh, Why, bh, by, epochs, learning_rate, save_weights=True):
        """
        Train the RNN using the DLL and save weights to a file after training.

        Parameters:
        - X: np.ndarray, token indices as float32 [sequence_length]
        - unique_tokens: list of str, unique tokens in the vocabulary
        - embeddings: np.ndarray, [vocab_size, embedding_dim]
        - Wxh: np.ndarray, [embedding_dim, hidden_size]
        - Whh: np.ndarray, [hidden_size, hidden_size]
        - Why: np.ndarray, [hidden_size, vocab_size]
        - bh: np.ndarray, [hidden_size]
        - by: np.ndarray, [vocab_size]
        - epochs: int, number of training epochs
        - learning_rate: float, learning rate for training
        - save_weights: bool, whether to save weights after training
        """
        # Set the vocabulary at the start of training
        self.set_vocabulary(unique_tokens)

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

        # Validate that vocab_size matches the number of unique tokens
        if vocab_size != len(unique_tokens):
            raise ValueError(f"Vocabulary size in embeddings ({vocab_size}) does not match length of unique_tokens ({len(unique_tokens)}).")

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
        
        # Store weights in the instance for later use (e.g., inference)
        self.embeddings = embeddings
        self.Wxh = Wxh
        self.Whh = Whh
        self.Why = Why
        self.bh = bh
        self.by = by

        if save_weights:
            # Save weights to a file, including vocabulary
            weights_file = f"rnn_weights{str(time())}.npz"
            print(f"Saving weights to '{weights_file}'...")
            try:
                np.savez(
                    weights_file,
                    embeddings=embeddings,
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

    def load_model(self, weights_file):
        """
        Load model weights and vocabulary from a file.

        Parameters:
        - weights_file: str, path to the .npz file containing the weights
        """
        data = np.load(weights_file, allow_pickle=True)
        self.embeddings = data['embeddings']
        self.Wxh = data['Wxh']
        self.Whh = data['Whh']
        self.Why = data['Why']
        self.bh = data['bh']
        self.by = data['by']
        # Load vocabulary if present
        if 'unique_tokens' in data:
            self.unique_tokens = data['unique_tokens'].tolist()
            self.token_to_idx = {token: idx for idx, token in enumerate(self.unique_tokens)}
            self.idx_to_token = {idx: token for idx, token in enumerate(self.unique_tokens)}
        else:
            print("Warning: No vocabulary found in the weights file. Call set_vocabulary separately if needed.")

    def inference(self, input_text, max_len=50):
        """
        Perform inference on text input using the DLL.

        Parameters:
        - input_text: str, input text to condition the generation
        - max_len: int, maximum length of the generated sequence

        Returns:
        - str, generated text
        """
        if not hasattr(self, 'unique_tokens'):
            raise ValueError("Vocabulary not set. Call set_vocabulary or load_model with vocabulary first.")
        if not hasattr(self, 'embeddings'):
            raise ValueError("Model weights not loaded. Call load_model or train first.")

        # Tokenize the input text (simple whitespace splitting)
        input_tokens = input_text.split()
        if not input_tokens:
            raise ValueError("Input text is empty.")
        # Use the last token as the starting index; 0 for unknown tokens
        start_idx = self.token_to_idx.get(input_tokens[-1], 0)

        # Get the end-of-sequence token index
        eos_idx = self.token_to_idx["<EOS>"]

        # Prepare the Params structure
        p = Params()
        p.vocab_size = len(self.unique_tokens)
        p.embedding_dim = self.embeddings.shape[1]
        p.hidden_size = self.Whh.shape[0]
        p.embeddings = self.embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p.Wxh = self.Wxh.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p.Whh = self.Whh.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p.Why = self.Why.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p.bh = self.bh.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p.by = self.by.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Call the generate function from the DLL
        gen_len = ctypes.c_int()
        generated_ptr = self.rnn_lib.generate(
            ctypes.byref(p),
            start_idx,
            eos_idx,
            max_len,
            ctypes.byref(gen_len)
        )

        # Check if generate returned NULL
        if not generated_ptr:
            print("Warning: generate returned NULL, no sequence generated.")
            return ""

        # Convert the generated sequence to a Python list
        generated = [int(generated_ptr[i]) for i in range(gen_len.value)]

        # Free the memory allocated in C using the DLL's function
        self.rnn_lib.free_generated(generated_ptr)

        # Convert indices back to tokens
        generated_tokens = [self.idx_to_token[idx] for idx in generated]

        # Join tokens into a string
        return ' '.join(generated_tokens)