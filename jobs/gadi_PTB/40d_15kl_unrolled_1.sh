#!/bin/bash
# Experiment Pipeline
# Robert Jeffrey <robert.jeffrey@anu.edu.au>

DEQ_DIR="${HOME}/deq/DEQModel"

eval "$(/home/659/rj4139/miniconda3/bin/conda shell.bash hook)"
conda activate honours

cd ${DEQ_DIR}
bash ${DEQ_DIR}/run_trellisnet_experiment.sh train --name 40d_15kl_unrolled_1.sh --n_layer 40 --pretrain_steps 15000 --seed 100 --timing --time_limit 16000

