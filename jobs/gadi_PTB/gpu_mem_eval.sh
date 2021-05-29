#!/bin/bash
# Experiment Pipeline
# Robert Jeffrey <robert.jeffrey@anu.edu.au>

DEQ_DIR="/g/data1a/ll21/deq/DEQModel"

eval "$(/home/659/rj4139/miniconda3/bin/conda shell.bash hook)"
conda activate honours

cd ${DEQ_DIR}

bash ${DEQ_DIR}/run_trellisnet_experiment.sh train --name mem_deq --n_layer 3 --pretrain_steps 0 --seed 9999 --timing --eval_mem
bash ${DEQ_DIR}/run_trellisnet_experiment.sh train --name mem_3l --n_layer 3 --pretrain_steps 15000 --seed 9999 --timing --eval_mem
bash ${DEQ_DIR}/run_trellisnet_experiment.sh train --name mem_20l --n_layer 20 --pretrain_steps 15000 --seed 9999 --timing --eval_mem
bash ${DEQ_DIR}/run_trellisnet_experiment.sh train --name mem_40l --n_layer 40 --pretrain_steps 15000 --seed 9999 --timing --eval_mem
