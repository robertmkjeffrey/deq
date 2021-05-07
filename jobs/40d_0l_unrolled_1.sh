#!/bin/bash
# Experiment Pipeline
# Robert Jeffrey <robert.jeffrey@anu.edu.au>

DEQ_DIR="${HOME}/deq/DEQModel"

conda activate honours

cd ${DEQ_DIR}
bash ${DEQ_DIR}/run_trellisnet_experiment.sh train --name 40d_0l_unrolled_1 --n_layer 40 --pretrain_steps 0 --seed 100 --time

