from datetime import datetime, timedelta
from textwrap import dedent

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# Operators; we need this to operate!
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from random import random, randrange, randint

from pyspark_cleaning import main
from hdfs.client import Client
from airflow.providers.apache.hdfs.sensors.hdfs import HdfsSensor
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.utils.dates import days_ago

import os

import logging

logger = logging.getLogger("airflow.task")


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
    description="Fraud transactions generation",
    schedule_interval=timedelta(minutes=10),
    start_date=days_ago(1),
    catchup=False,
    tags=["homework_4"],
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

    # hdfs_file_created = HdfsSensor(
    #    task_id='check_if_file_is_generated',
    #    filepath=f'/user/airflow/input_files/partition_{fileCount-2}.parquet',
    #    hdfs_conn_id='dr.who',
    #    poke_interval=120,
    #    timeout=240
    # )

    spark_submit_cluster = SparkSubmitOperator(
        application="/home/ubuntu/fraud-detection-ml/Homework_4/dags/pyspark_cleaning.py",
        task_id="spark_submit_task",
    )

    # hdfs_file_created >> spark_submit_cluster
    spark_submit_cluster
