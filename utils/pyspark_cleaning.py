# df
import findspark

findspark.init()
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder, MinMaxScaler
from hdfs.client import Client
from pyspark.sql import SparkSession
import sys
import logging
import os
import pyspark.sql.functions as F
from pyspark.ml import Pipeline

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
import mlflow

os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
TRACKING_SERVER_HOST = os.getenv(
    "TRACKING_SERVER_HOST"
)  # fill in with the public DNS of the EC2 instance
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5001")
mlflow.set_experiment("fraud-transaction-prediction")

logger = logging.getLogger(__name__)


def clean_data(**kwargs):

    ti = kwargs["ti"]
    partition = ti.xcom_pull(task_ids="combine_data", key="batch_to_train")

    file_name = f"/user/airflow/intermediate_files/{partition}"

    spark = SparkSession.builder.master("yarn").getOrCreate()
    df = spark.read.format("parquet").load(file_name)

    numericColumns = list(
        map(
            lambda x: x[0],
            filter(lambda x: x[1] == "double" or x[1] == "int", df.dtypes),
        )
    )
    stringColumns = ["TERMINAL_ID"]
    numericAssembler = (
        VectorAssembler().setInputCols(numericColumns).setOutputCol("features")
    )

    stringColumnsIndexed = list(map(lambda x: x + "_Indexed", stringColumns))

    indexer = (
        StringIndexer().setInputCols(stringColumns).setOutputCols(stringColumnsIndexed)
    )

    catColumns = list(map(lambda x: x + "_Coded", stringColumnsIndexed))

    encoder = (
        OneHotEncoder().setInputCols(stringColumnsIndexed).setOutputCols(catColumns)
    )

    featureColumns = ["scaledFeatures"] + catColumns

    scaler = MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")

    assembler = (
        VectorAssembler().setInputCols(featureColumns).setOutputCol("featuresFinal")
    )

    pipeline = Pipeline().setStages(
        [numericAssembler, indexer, encoder, scaler, assembler]
    )
    pipelined_model = pipeline.fit(df)
    result = pipelined_model.transform(df).select("featuresFinal", "TX_FRAUD")
    result = result.withColumn("TX_FRAUD_I", result["TX_FRAUD"].cast("integer"))
    labelIndexer = StringIndexer().setInputCol("TX_FRAUD_I").setOutputCol("TX_FRAUD")
    model_label_indexer = labelIndexer.fit(result.select("TX_FRAUD_I", "featuresFinal"))
    result = model_label_indexer.transform(result.select("TX_FRAUD_I", "featuresFinal"))

    result.select("*").orderBy(F.rand()).write.format("parquet").mode("overwrite").save(
        f"/user/airflow/processed_files/{partition}"
    )
    kwargs["ti"].xcom_push(
        key="data_path",
        value=f"/user/airflow/processed_files/{partition}",
    )

    with mlflow.start_run() as active_run:
        run_id = active_run.info.run_id
        mlflow.spark.log_model(pipelined_model, "pipeline")
        kwargs["ti"].xcom_push(
            key="run_id_with_pipeline",
            value=str(run_id),
        )


if __name__ == "__main__":
    clean_data()
