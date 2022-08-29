FROM apache/airflow:latest
USER root
RUN usermod -G root airflow
RUN sudo apt update && sudo apt install -y heimdal-dev && sudo apt install -y build-essential && sudo apt install default-jdk
RUN export JAVA_HOME='/usr/lib/jvm/java-8-openjdk-amd64'
RUN export PATH=$PATH:$JAVA_HOME/bin
RUN chmod 777 /usr/bin/gcc
USER airflow
#RUN python -m pip install --upgrade pip
#RUN pip install hdfs pyspark apache-airflow-providers-apache-hdfs