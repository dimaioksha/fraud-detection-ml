from cmath import log
import findspark

findspark.init()
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder, MinMaxScaler
from hdfs.client import Client
from pyspark.sql import SparkSession
import sys
import os
import logging
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

import os
from dotenv import load_dotenv
import mlflow

load_dotenv("/home/ubuntu/fraud-detection-ml/.env")

# os.environ["AWS_PROFILE"] = "" # fill in with your AWS profile. More info: https://docs.aws.amazon.com/sdk-for-java/latest/developer-guide/setup.html#setup-credentials
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
TRACKING_SERVER_HOST = os.getenv(
    "TRACKING_SERVER_HOST"
)  # fill in with the public DNS of the EC2 instance
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5001")
mlflow.set_experiment("fraud-transaction-prediction")

logger = logging.getLogger(__name__)


def train_model(**kwargs):
    mlflow.pyspark.ml.autolog()

    logger.info("GOT INTO TRAIN FUNCTION")
    ti = kwargs["ti"]
    name_of_file = ti.xcom_pull(task_ids="clean_data", key="data_path")

    spark = SparkSession.builder.appName("train_model").master("yarn").getOrCreate()

    df = spark.read.format("parquet").load(name_of_file)

    splits = df.randomSplit([0.7, 0.3])
    train_df = splits[0]
    test_df = splits[1]

    with mlflow.start_run():

        lr = LogisticRegression(
            featuresCol="featuresFinal",
            labelCol="TX_FRAUD",
            maxIter=10,
            regParam=0.3,
            elasticNetParam=0.8,
        )

        lr_model = lr.fit(train_df)

        predictions = lr_model.transform(test_df)

        evaluator = BinaryClassificationEvaluator().setLabelCol("TX_FRAUD")
        roc_auc = evaluator.evaluate(predictions)

        mlflow.log_metric("accuracy", roc_auc)

        logger.info(f"Accuracy SCORE: {roc_auc}")


if __name__ == "__main__":
    train_model()
