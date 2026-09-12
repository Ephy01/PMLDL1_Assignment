FROM python:3.12-slim-bookworm
WORKDIR /app
COPY code/deployment/app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY code/deployment/app/app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
