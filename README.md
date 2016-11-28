## A custom CNN for the DREAM ENCODE challenge

This is a pretty standard convolutional neural net on genomic sequence with the following features added:
* normalized per-base DNase I cuts for the + and - strand are concatenated onto the one hot encoding of sequence, to give a [sequence context] x 6 input matrix.
* gene expression PCs are included as features to allow the model to interpolate between different cell types.
* a three class ordinal likelihood is used for the Unbound/Ambiguous/Bound labels.
* simultaneous analysis of the forward and reverse complement.
* down-sampling of the negative set to speed up training (and accounting for by weighting the likelihood). 

From the round 2 leaderboard you can see performance is highly competitive for some TFs (e.g. MAX https://www.synapse.org/#!Synapse:syn6131484/wiki/402503) and less so for others (e.g. REST https://www.synapse.org/#!Synapse:syn6131484/wiki/402505).

The repo is intended to be fully self contained (save dependencies on synapseclient, pysam and pyDNase python packages), including programmatic download of challenge data, pre-processing, model fitting, prediction and submission.

### Installation

You'll need the following python packages: pysam, pyDNase, scikit-learn (for performance metrics), synapseclient (for downloading the data and submitting), numpy, scipy, theano. 

### Usage

The script `run_all.sh` will in principle run all of these steps for you. Realistically you'll want to train each TF model (and probably do the DNase pre-processing) on a cluster since this is pretty time consuming (10ish hours). 

1. Set a data location, e.g. add something like the following to your ~/.bash_profile
```
export DREAM_ENCODE_DATADIR=/myscratchspace/dream_encode/
```

2. Download the challenge data using `download_challenge_data.py`, but note you'll need to set your Synapse email/password in that script.

3. [optional] Calculate gene expression PCs using `gene_expression_pca.R`. I included the output file, 'ge_pca.txt' so you don't strictly need to rerun this. If you do want to do this yourself you'll need the R packages irlba and foreach.

4. Calculate DNase I cut counts using the `get_DNase_cuts.py` script. This converts the DNase I bams into an efficient numpy representation of cut counts saved in .npz files. The bam first need indexing (e.g. using samtools). `index.sh` will do this for you. 

5. Train models for each TF using `train.py`. This script includes outputting leaderboard and final submissions.

6. Submit to Synapse using `submit.py`. Note you'll need to set up a folder in Synapse to use for this and set the id in the script. 