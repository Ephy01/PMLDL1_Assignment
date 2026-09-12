from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PYTHON = sys.executable  

with DAG(
    dag_id="ford_price_pipeline",
    description="Ford advertised-price model: prepare data, train, deploy",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 9, 1),
    catchup=False,
    is_paused_upon_creation=True,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=1)},
    tags=["pmldl", "mlops"],
) as dag:
    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=f"cd '{PROJECT_ROOT}' && {PYTHON} code/datasets/prepare_data.py",
    )
    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"cd '{PROJECT_ROOT}' && {PYTHON} code/models/train_model.py",
    )
    deploy = BashOperator(
        task_id="deploy",
        bash_command=(
            f"cd '{PROJECT_ROOT}/code/deployment' && "
            "docker compose up --build -d --remove-orphans"
        ),
        env={"PATH": "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin"},
        append_env=True,
    )


    prepare_data >> train_model >> deploy
