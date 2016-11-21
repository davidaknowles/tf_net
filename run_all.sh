#!/bin/bash

if [ -z "$DREAM_ENCODE_DATADIR" ]; then
    echo "Please set (e.g. in your .bash_profile) DREAM_ENCODE_DATADIR to where you want challenge data stored."
    exit 1
fi

echo "Downloading challenge data"
python2.7 download_challenge_data.py

echo "Calculating gene expression PCs"
Rscript gene_expression_pca.R

echo "Converting DNase BAMs to read cuts"
for SLURM_ARRAY_TASK_ID in {0..13}; do
    export SLURM_ARRAY_TASK_ID
    python2.7 get_DNase_cuts.py
done

echo "Training models and predicting"
for SLURM_ARRAY_TASK_ID in {0..12}; do
    export SLURM_ARRAY_TASK_ID
    python2.7 train.py
done

echo "Submitting"
python2.7 submit.py
			  
