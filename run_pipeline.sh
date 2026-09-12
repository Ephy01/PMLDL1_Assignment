set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-.venv/bin/python}
$PY code/datasets/prepare_data.py
$PY code/models/train_model.py
(cd code/deployment && docker compose up --build -d --remove-orphans)
echo "API:  http://localhost:8000/docs"
echo "App:  http://localhost:8501"
