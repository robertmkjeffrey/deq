#!/bin/bash
# Experiment Pipeline
# Robert Jeffrey <robert.jeffrey@anu.edu.au>

DEQ_DIR="/g/data1a/ll21/deq/DEQModel"

eval "$(/home/659/rj4139/miniconda3/bin/conda shell.bash hook)"
conda activate honours

cd ${DEQ_DIR}
bash ${DEQ_DIR}/run_trellisnet_experiment.sh train --name mem_sat_40kl --n_layer 40 --pretrain_steps 5000 --seed 500 --timing --time_limit 16000 --batch_size 32 --gpu0_bsz -1
