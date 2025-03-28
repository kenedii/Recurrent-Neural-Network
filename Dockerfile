FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Set a writable temporary directory for pip
ENV TMPDIR=/app/tmp
RUN mkdir -p /app/tmp

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY api.py .
COPY rnn.py .
COPY torch_rnn.py .
COPY weights/ ./weights/
COPY lib/rnnlib.so .  

# Clean up temporary directory
RUN rm -rf /app/tmp

EXPOSE 8080

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]