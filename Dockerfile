FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
COPY api.py .
COPY rnn.py .
COPY torch_rnn.py .
COPY weights/ ./weights/
COPY rnn.so .

RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x rnn.so

EXPOSE 8080

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]