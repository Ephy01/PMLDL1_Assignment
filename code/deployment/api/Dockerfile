FROM python:3.12-slim-bookworm
WORKDIR /app
COPY code/deployment/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY code/deployment/api/main.py .
COPY models/model.onnx models/feature_meta.json /models/
ENV MODEL_DIR=/models
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
