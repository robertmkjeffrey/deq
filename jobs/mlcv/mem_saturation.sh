GPUS="0 4"

cd ~/deq/DEQModel

bash run_trellisnet_experiment.sh train --name mem_deq --n_layer 3 --pretrain_steps 0 --seed 9999 --timing --eval_mem --use_gpus ${GPUS} --batch_size 32 --gpu0_bsz -1
bash run_trellisnet_experiment.sh train --name mem_3l --n_layer 3 --pretrain_steps 15000 --seed 9999 --timing --eval_mem --use_gpus ${GPUS} --batch_size 136 --gpu0_bsz -1
bash run_trellisnet_experiment.sh train --name mem_20l --n_layer 20 --pretrain_steps 15000 --seed 9999 --timing --eval_mem --use_gpus ${GPUS} --batch_size 48 --gpu0_bsz -1
bash run_trellisnet_experiment.sh train --name mem_40l --n_layer 40 --pretrain_steps 15000 --seed 9999 --timing --eval_mem --use_gpus ${GPUS} --batch_size 26 --gpu0_bsz -1
