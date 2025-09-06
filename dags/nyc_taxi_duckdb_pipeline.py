from __future__ import annotations
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import timedelta

# This DAG orchestrates the NYC Taxi pipeline using project scripts in /opt/nyc (mounted volume).

DEFAULT_ARGS = {
    "owner": "data-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="nyc_taxi_duckdb_pipeline",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,  
    catchup=False,
    description="Download -> Ingest DuckDB -> Build Features -> Train Model",
    tags=["nyc", "duckdb", "ml"],
    params={
        "year": 2023,
        "month": 1,
        "h3_res": 8,
        "limit": None,  
    },
) as dag:

    download = BashOperator(
        task_id="download_month",
        bash_command=(
            "python -m src.ingest --year {{ params.year }} --month {{ params.month }} "
            + "{% if params.limit %}--limit {{ params.limit }}{% endif %}"
        ),
    )

    ingest_duckdb = BashOperator(
        task_id="ingest_duckdb",
        bash_command=(
            "python -m src.ingest_duckdb "
            "--db_path /opt/airflow/data/nyc_taxi.duckdb --raw_dir /opt/airflow/data/raw "
            "--year {{ params.year }} --month {{ params.month }}"
        ),
    )

    features_duckdb = BashOperator(
        task_id="build_features_duckdb",
        bash_command=(
            "MONTH_PADDED=$(printf '%02d' {{ params.month }}) && "
            "python -m src.features_duckdb "
            "--db_path /opt/airflow/data/nyc_taxi.duckdb "
            "--year {{ params.year }} --month {{ params.month }} "
            "--h3_res {{ params.h3_res }} "
            "--out /opt/airflow/data/features/train_{{ params.year }}_${MONTH_PADDED}.parquet"
        ),
    )

    train = BashOperator(
        task_id="train_model",
        bash_command=(
            "set -euo pipefail; "
            "MONTH_PADDED=$(printf '%02d' {{ params.month }}) && "
            "OUT=/opt/airflow/data/features/train_{{ params.year }}_${MONTH_PADDED}.parquet && "
            "echo \"[train] Je cherche: $OUT\" && "
            "[ -f \"$OUT\" ] || { echo '[train] Fichier manquant. Contenu:'; ls -lah /opt/airflow/data/features || true; exit 1; } && "
            "python -m src.train "
            "--train_path \"$OUT\" "
            "--model_path /opt/airflow/artifacts/model_rf.pkl "
            "--split time "
            "--cutoff_year 2023 "
            "--cutoff_month {{ params.month }} "
            "--cutoff_day 25 "
            "--sample_rows 150000 "
            "--n_estimators 200"
        ),
    )



    download >> ingest_duckdb >> features_duckdb >> train
