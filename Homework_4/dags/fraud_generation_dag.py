from datetime import datetime, timedelta
from textwrap import dedent

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# Operators; we need this to operate!
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from random import random, randrange, randint

from script_generation import main


with DAG(
    "fraud_generation",
    # These args will get passed on to each operator
    # You can override them on a per-task basis during operator initialization
    default_args={
        "depends_on_past": False,
        "email": ["airflow@example.com"],
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        # 'queue': 'bash_queue',
        # 'pool': 'backfill',
        # 'priority_weight': 10,
        # 'end_date': datetime(2016, 1, 1),
        # 'wait_for_downstream': False,
        # 'sla': timedelta(hours=2),
        # 'execution_timeout': timedelta(seconds=300),
        # 'on_failure_callback': some_function,
        # 'on_success_callback': some_other_function,
        # 'on_retry_callback': another_function,
        # 'sla_miss_callback': yet_another_function,
        # 'trigger_rule': 'all_success'
    },
    description="Fraud transactions generation",
    schedule_interval=timedelta(minutes=10),
    start_date=datetime(2022, 7, 5),
    catchup=False,
    tags=["homework_4"],
) as dag:

    _c_line_arg = randint(0, 32756)
    t2 = BashOperator(
        task_id="generate_transaction",
        bash_command=f"python3 /home/ubuntu/fraud-detection-ml/Homework_4/dags/script_generation.py {_c_line_arg}",
    )
    t2
