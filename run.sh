#!/bin/bash
source activate intel
export MKL_DYNAMIC=FALSE
export MKL_NUM_THREADS=5
rm -r output
mkdir output
mkdir output/data
mkdir output/figures
python src/main.py

