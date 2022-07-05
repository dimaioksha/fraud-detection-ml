#!/bin/bash
eval $(conda shell.bash hook)
conda activate main
for i in {3..20}
do
   python script_generation.py $i
done