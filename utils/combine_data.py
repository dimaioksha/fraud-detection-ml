from hdfs.client import Client
from pyspark.sql import SparkSession
import os


def combine_data(**kwargs):
    hdfs_url = os.getenv(
        "HDFS_NAMENODE_URL",
        "http://rc1b-dataproc-m-xzxwmqcudfo0foh0.mdb.yandexcloud.net:9870",
    )
    client = Client(hdfs_url)
    spark = SparkSession.builder.appName("union_data").getOrCreate()

    file_count = client.content("/user/airflow/input_files")["fileCount"]

    # считаем что мы накапливаем n-ное число входных батчей как тестовую дату
    if file_count > 10:
        start_file = file_count - 1 - 10
        end_file = file_count - 1
    else:
        start_file = 0
        end_file = file_count - 1

    result_df = None
    for index in range(start_file, end_file):
        if result_df is None:
            result_df = spark.read.format("parquet").load(
                f"/user/airflow/input_files/partition_{index}.parquet"
            )
        else:
            result_df = result_df.union(
                spark.read.format("parquet").load(
                    f"/user/airflow/input_files/partition_{index}.parquet"
                )
            )

    path = f"/user/airflow/intermediate_files/slice_of_{start_file}_{end_file}.parquet"
    result_df.write.format("parquet").mode("overwrite").save(path)

    kwargs["ti"].xcom_push(
        key="batch_to_train", value=f"slice_of_{start_file}_{end_file}.parquet"
    )


if __name__ == "__main__":
    combine_data()
