#!/bin/bash
# Experiment Pipeline
# Robert Jeffrey <robert.jeffrey@anu.edu.au>

DEQ_DIR="${HOME}/deq/DEQModel"

eval "$(/home/659/rj4139/miniconda3/bin/conda shell.bash hook)"
conda activate honours

cd ${DEQ_DIR}
bash ${DEQ_DIR}/run_trellisnet_experiment.sh train --name deq_1 --n_layer 40 --pretrain_steps 0 --seed 100 --timing

