import os
import glob
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import rnn  # Import the RNNTrainer class from rnn.py

# Initialize the FastAPI app
app = FastAPI()

# Determine the weights file to use
WEIGHTS_FILE = os.getenv("WEIGHTS_FILE")
if WEIGHTS_FILE is None:
    # Look for all weights files matching the pattern 'rnn_weights*.npz'
    weights_files = glob.glob("rnn_weights*.npz")
    # Filter to include only files with a numeric timestamp (e.g., rnn_weights1742782852.4370303.npz)
    weights_files = [f for f in weights_files if re.match(r'rnn_weights\d+\.\d+\.npz', f)]
    if weights_files:
        # Select the file with the latest timestamp
        WEIGHTS_FILE = max(weights_files, key=lambda x: float(x.split('rnn_weights')[1].split('.npz')[0]))
    else:
        raise FileNotFoundError("No valid weights files found matching 'rnn_weights<timestamp>.npz'")
else:
    # If WEIGHTS_FILE is specified via environment variable, ensure it exists
    if not os.path.exists(WEIGHTS_FILE):
        raise FileNotFoundError(f"Weights file {WEIGHTS_FILE} not found")

print(f"Using weights file: {WEIGHTS_FILE}")

# Define the path to the RNN DLL
DLL_PATH = os.path.join(os.path.dirname(__file__), "src", "rnn.dll")

# Initialize and load the RNNTrainer with the pre-trained model
try:
    rnn_trainer = rnn.RNNTrainer(dll_path=DLL_PATH)
    rnn_trainer.load_model(WEIGHTS_FILE)
    # Check that unique_tokens is loaded, as required by the weights format
    if not hasattr(rnn_trainer, 'unique_tokens'):
        raise RuntimeError("Loaded model does not contain 'unique_tokens'. Cannot perform inference.")
    print(f"Model loaded successfully from {WEIGHTS_FILE}")
except Exception as e:
    raise RuntimeError(f"Failed to load model from {WEIGHTS_FILE}: {str(e)}")

# Define the request model for the inference endpoint
class InferenceRequest(BaseModel):
    input_text: str
    max_len: int = 50  # Default maximum length of generated text

# Define the inference endpoint
@app.post("/inference/", response_model=dict)
async def perform_inference(request: InferenceRequest):
    """
    Perform inference using the loaded RNN model.

    Args:
        request: InferenceRequest object containing input_text and max_len

    Returns:
        dict: JSON response with the generated text

    Raises:
        HTTPException: If inference fails due to invalid input or internal errors
    """
    try:
        generated_text = rnn_trainer.inference(request.input_text, request.max_len)
        return {"generated_text": generated_text}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

# health check endpoint
@app.get("/health/", response_model=dict)
async def health_check():
    """
    Check the health status of the API.

    Returns:
        dict: JSON response indicating the API status
    """
    return {"status": "healthy"}

# Run the app with: uvicorn api:app --reload --port=8888
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)