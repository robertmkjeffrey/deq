#!/bin/bash
    # Experiment Pipeline
    # Robert Jeffrey <robert.jeffrey@anu.edu.au>

    DEQ_DIR="/g/data1a/ll21/deq/DEQModel"

    eval "$(/home/659/rj4139/miniconda3/bin/conda shell.bash hook)"
    conda activate honours

    cd ${DEQ_DIR}
    bash ${DEQ_DIR}/run_trellisnet_experiment.sh train --name convergence_20d_20kl_2 --n_layer 20 --pretrain_steps 20000 --seed 200 --epochs 82 --force-deq-validation
    