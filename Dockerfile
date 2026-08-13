FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# CPU-only torch first: sentence-transformers otherwise pulls the CUDA build
# (~1GB of nvidia/cudnn wheels) even though this container has no GPU.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

COPY app ./app
COPY scripts ./scripts

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
