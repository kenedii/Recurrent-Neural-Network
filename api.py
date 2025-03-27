import os
import torch
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel
import rnn  # Import RNNTrainer from rnn.py
from torch_rnn import RNNModel, generate  # Import RNNModel and generate from torch_rnn.py

# Initialize the FastAPI app
app = FastAPI()

# Define the directory where weights files are stored
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
# Ensure the weights directory exists
if not os.path.exists(WEIGHTS_DIR):
    os.makedirs(WEIGHTS_DIR)

# Define the path to the RNN DLL
DLL_PATH = os.path.join(PROJECT_ROOT, "rnn.dll")

# Define the request model for inference endpoints
class InferenceRequest(BaseModel):
    input_text: str
    max_len: int = 50  # Default maximum length of generated text
    weights_filename: str  # Filename of the weights file to use

# Endpoint for inference using RNNTrainer (DLL-based model)
@app.post("/inference_rnn_from_weights/")
async def inference_rnn_from_weights(request: InferenceRequest):
    """
    Perform inference using the RNNTrainer model loaded from the specified weights file.

    Args:
        request: InferenceRequest object containing input_text, max_len, and weights_filename

    Returns:
        dict: JSON response with the generated text

    Raises:
        HTTPException: If the weights file is invalid, not found, or inference fails
    """
    weights_path = os.path.abspath(os.path.join(WEIGHTS_DIR, request.weights_filename))
    if not weights_path.startswith(WEIGHTS_DIR):
        raise HTTPException(status_code=400, detail="Invalid weights filename")
    if not os.path.exists(weights_path):
        raise HTTPException(status_code=404, detail="Weights file not found")
    try:
        trainer = rnn.RNNTrainer(dll_path=DLL_PATH)
        trainer.load_model(weights_path)
        generated_text = trainer.inference(request.input_text, request.max_len)
        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

# Endpoint for inference using PyTorch RNN model
@app.post("/inference_rnn_from_pytorch_weights/")
async def inference_rnn_from_pytorch_weights(request: InferenceRequest):
    """
    Perform inference using the PyTorch RNN model loaded from the specified weights file.

    Args:
        request: InferenceRequest object containing input_text, max_len, and weights_filename

    Returns:
        dict: JSON response with the generated text

    Raises:
        HTTPException: If the weights file is invalid, not found, or inference fails
    """
    weights_path = os.path.abspath(os.path.join(WEIGHTS_DIR, request.weights_filename))
    if not weights_path.startswith(WEIGHTS_DIR):
        raise HTTPException(status_code=400, detail="Invalid weights filename")
    if not os.path.exists(weights_path):
        raise HTTPException(status_code=404, detail="Weights file not found")
    try:
        saved_data = torch.load(weights_path)
        unique_tokens = saved_data['unique_tokens']
        token_to_idx = saved_data['token_to_idx']
        idx_to_token = {idx: token for token, idx in token_to_idx.items()}
        vocab_size = len(unique_tokens)
        embedding_dim = 300  # Matches Word2Vec embedding size in torch_rnn.py
        hidden_size = 100  # Matches hidden_size in torch_rnn.py
        # Create a dummy embedding matrix (overwritten by state dict)
        embedding_matrix = torch.zeros((vocab_size, embedding_dim), dtype=torch.float32)
        model = RNNModel(vocab_size, embedding_dim, hidden_size, embedding_matrix)
        model.load_state_dict(saved_data['model_state_dict'])
        model.eval()
        generated_text = generate(model, request.input_text, token_to_idx, idx_to_token, request.max_len)
        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

# Endpoint for uploading weights files
@app.post("/upload_weights/")
async def upload_weights(file: UploadFile):
    """
    Upload a weights file to the server and store it in the weights folder.

    Args:
        file: The weights file to upload

    Returns:
        dict: JSON response with the filename of the uploaded file

    Raises:
        HTTPException: If the file upload fails
    """
    try:
        file_path = os.path.join(WEIGHTS_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        return {"filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

# Health check endpoint
@app.get("/health/")
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