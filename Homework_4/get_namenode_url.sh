#!/bin/bash
address="$(hdfs getconf -confKey fs.defaultFS)"
local_address="$(echo "$(hdfs getconf -confKey dfs.namenode.http-address)" | rev)"
port_number="$(echo "${local_address:0:4}" | rev)"
echo "http://${address:7}:${port_number}"
