bash run_trellisnet_experiment.sh train --name mem_deq --n_layer 3 --pretrain_steps 0 --seed 9999 --timing --eval_mem
bash run_trellisnet_experiment.sh train --name mem_3l --n_layer 3 --pretrain_steps 15000 --seed 9999 --timing --eval_mem
bash run_trellisnet_experiment.sh train --name mem_20l --n_layer 20 --pretrain_steps 15000 --seed 9999 --timing --eval_mem
bash run_trellisnet_experiment.sh train --name mem_40l --n_layer 40 --pretrain_steps 15000 --seed 9999 --timing --eval_mem
