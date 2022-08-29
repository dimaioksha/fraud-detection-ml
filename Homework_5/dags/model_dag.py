from datetime import datetime, timedelta
from textwrap import dedent

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# Operators; we need this to operate!
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from random import random, randrange, randint

from utils.pyspark_cleaning import main
from hdfs.client import Client
from airflow.providers.apache.hdfs.sensors.hdfs import HdfsSensor
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.utils.dates import days_ago
from utils.train_model import train_model

import os

import logging

logger = logging.getLogger("airflow.task")

class SparkSubmitOperatorXCom(SparkSubmitOperator):

    def execute(self, context):
        super().execute(context)
        return self._hook._driver_status

with DAG(
    "preprocessing_data",
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
    description="MODEL_LIFECYCLE",
    schedule_interval=timedelta(minutes=10),
    start_date=days_ago(1),
    catchup=False,
    tags=["homework_5"],
) as dag:

    hdfs_url = os.getenv(
        "HDFS_NAMENODE_URL",
        "http://rc1b-dataproc-m-xzxwmqcudfo0foh0.mdb.yandexcloud.net:9870",
    )
    client = Client(hdfs_url)

    fileCount = client.content("/user/airflow/input_files")["fileCount"]

    if fileCount == 0:
        fileCount += 1

    logger.info(f"Partition: /user/airflow/input_files/partition_{fileCount-1}.parquet")

    clean_data = SparkSubmitOperatorXCom(
        application="/home/ubuntu/fraud-detection-ml/utils/pyspark_cleaning.py",
        task_id="clean_data",
        do_xcom_push=True
    )

    fit_model = PythonOperator(
        python_callable=train_model,
        task_id="fit_model",
        provide_context=True
    )

    clean_data >> fit_model








