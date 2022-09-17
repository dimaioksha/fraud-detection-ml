import os
from dotenv import load_dotenv
import mlflow
from pyspark.sql import SparkSession
from mlflow.tracking import MlflowClient
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import logging
import pandas as pd
import numpy as np
from ._f1_binary_evaluator import F1BinaryEvaluator
from sklearn.metrics import f1_score as f1_score_calc
from scipy.stats import ttest_ind

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


def _get_bootstraped_metric_for_model(model, transformed_test_df):
    np.random.seed(42)
    predictions = model.transform(transformed_test_df)
    f1_evaluator = F1BinaryEvaluator(labelCol="TX_FRAUD")
    f1_score = f1_evaluator.evaluate(predictions)
    pandas_predictions = predictions.toPandas()
    y_true = transformed_test_df.select("TX_FRAUD").toPandas()
    bootstrap_iterations = 1000
    df_bootstrap = pd.DataFrame(
        {"y_test": y_true["TX_FRAUD"], "y_pred": pandas_predictions["prediction"]}
    )
    y_true_mean = y_true["TX_FRAUD"].mean()
    y_pred_mean = pandas_predictions["prediction"].mean()

    logger.info(f"y_test mean {y_true_mean}")
    logger.info(f"y_pred mean {y_pred_mean}")

    scores = pd.DataFrame(data={"F1": 0.0}, index=range(bootstrap_iterations))

    for i in range(bootstrap_iterations):
        sample = df_bootstrap.sample(frac=1.0, replace=True)
        scores.loc[i, "F1"] = f1_score_calc(sample["y_test"], sample["y_pred"])
    logger.info(f"POINT ESTIMATION OF F1 IS {f1_score}")
    return scores


def _preprocess_with_pipeline(pipeline, raw):
    transformed_test_df = pipeline.transform(raw).select("featuresFinal", "TX_FRAUD")
    transformed_test_df = transformed_test_df.withColumn(
        "TX_FRAUD", transformed_test_df["TX_FRAUD"].cast("integer")
    )
    return transformed_test_df


def validate_model(**kwargs):
    spark = SparkSession.builder.appName("validate_model").master("yarn").getOrCreate()
    client = MlflowClient(f"http://{TRACKING_SERVER_HOST}:5001")

    ti = kwargs["ti"]
    run_id = ti.xcom_pull(task_ids="clean_data", key="run_id_with_pipeline")
    partition = ti.xcom_pull(task_ids="generate_data", key="last_index_generated_file")
    full_path_new_portion = f"/user/airflow/input_files/partition_{partition}.parquet"
    test_df = spark.read.format("parquet").load(full_path_new_portion)
    model = mlflow.spark.load_model(f"runs:/{run_id}/model")
    pipeline = mlflow.spark.load_model(f"runs:/{run_id}/pipeline")

    transformed_test_df = _preprocess_with_pipeline(pipeline, test_df)

    current_model_metric = _get_bootstraped_metric_for_model(model, transformed_test_df)

    best_model = mlflow.spark.load_model(
        "models:/fraud-transaction-predictor/Production"
    )
    best_pipeline = mlflow.spark.load_model(
        "models:/fraud-transaction-pipeline/Production"
    )

    transformed_test_df_best_model = _preprocess_with_pipeline(best_pipeline, test_df)
    best_model_metric = _get_bootstraped_metric_for_model(
        best_model, transformed_test_df_best_model
    )

    pvalue = ttest_ind(
        best_model_metric["F1"], current_model_metric["F1"], alternative="less"
    ).pvalue
    mean_cur = current_model_metric["F1"].mean()
    std_cur = current_model_metric["F1"].std()
    mean_best = best_model_metric["F1"].mean()
    std_best = best_model_metric["F1"].std()
    logger.info(f"Mean {mean_cur} and std of considering model {std_cur}")
    logger.info(f"Mean {mean_best} and std of best model {std_best}")
    if pvalue < 0.05:
        # update model
        logger.info("NEW MODEL IS BETTER")
        mlflow.register_model(
            model_uri=f"runs:/{run_id}/model", name="fraud-transaction-predictor"
        )
        mlflow.register_model(
            model_uri=f"runs:/{run_id}/pipeline", name="fraud-transaction-pipeline"
        )
        client.transition_model_version_stage(
            name="fraud-transaction-predictor",
            version=client.get_latest_versions("fraud-transaction-predictor")[
                -1
            ].version,
            stage="Production",
            archive_existing_versions=True,
        )
        client.transition_model_version_stage(
            name="fraud-transaction-pipeline",
            version=client.get_latest_versions("fraud-transaction-pipeline")[
                -1
            ].version,
            stage="Production",
            archive_existing_versions=True,
        )

    else:
        logger.info("NEW MODEL IS WORSE THEN THE PRODUCTION")


if __name__ == "__main__":
    validate_model()
