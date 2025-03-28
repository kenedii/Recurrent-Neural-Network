# Recurrent-Neural-Network
RNN implemented in C with Python FastAPI endpoints

Instructions:
pip install -r requirements.txt

Windows:

Training:

C RNN: `python train.py`

Pytorch: `python torch_rnn.py`

Inference:

`uvicorn api:app --reload --port=`

Docker/Linux:

Training: 

C RNN: Ensure "dll path" in main.py points to `lib/rnnlib.so`

`python train.py`

Pytorch: `python torch_rnn.py`

Inference:

Ensure "dll path" in api.py points to `lib/rnnlib.so`

`uvicorn api:app --reload --port=`
