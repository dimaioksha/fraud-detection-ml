# df
import findspark

findspark.init()
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder, MinMaxScaler
from hdfs.client import Client
from pyspark.sql import SparkSession
import sys
import os


def main():
    hdfs_url = os.getenv(
        "HDFS_NAMENODE_URL",
        "http://rc1b-dataproc-m-xzxwmqcudfo0foh0.mdb.yandexcloud.net:9870",
    )
    client = Client(hdfs_url)

    partition = client.content("/user/airflow/input_files")["fileCount"] - 1

    if partition == 0:
        partition += 1

    file_name = f"/user/airflow/input_files/partition_{partition}.parquet"
    target = "TX_FRAUD"
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

    numeric = numericAssembler.transform(df)

    stringColumnsIndexed = list(map(lambda x: x + "_Indexed", stringColumns))

    indexer = (
        StringIndexer().setInputCols(stringColumns).setOutputCols(stringColumnsIndexed)
    )

    indexed = indexer.fit(numeric).transform(numeric)

    catColumns = list(map(lambda x: x + "_Coded", stringColumnsIndexed))

    encoder = (
        OneHotEncoder().setInputCols(stringColumnsIndexed).setOutputCols(catColumns)
    )

    encoded = encoder.fit(indexed).transform(indexed)

    featureColumns = ["scaledFeatures"] + catColumns

    scaler = MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")

    scaled = scaler.fit(encoded).transform(encoded)

    assembler = (
        VectorAssembler().setInputCols(featureColumns).setOutputCol("featuresFinal")
    )

    result = assembler.transform(scaled).select("featuresFinal", "TX_FRAUD")

    result.write.format("parquet").mode("overwrite").save(
        f"/user/airflow/processed_files/partition_{partition}.parquet"
    )


if __name__ == "__main__":
    main()
