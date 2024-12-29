#!/bin/bash
#SBATCH -p batch
#SBATCH -t 5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=80G
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --job-name=pytorch

shopt -s expand_aliases
source ~/.bash_aliases

myconda; conda activate torchmd-net

mkdir -p output

#torchmd-train --conf TensorNet-rMD17.yaml --log-dir output --coord-files qm_coord.npy --energy-files energy.npy --force-files qm_force.npy --embed-files qm_type.npy --ext-charge-coord-files mm_coord.npy --ext-charge-files mm_charge.npy --ext-esp-files mm_esp.npy --ext-esp-grad-files mm_esp_grad.npy --wandb-use True --wandb-name tensornet-cas9 --wandb-project cas9

export WANDB_API_KEY="fe09fab4519e7d2350e830048fea3b353aef36fe"

torchmd-train \
	--conf TensorNet-rMD17.yaml \
	--log-dir output \
	--coord-files qm_coord.npy \
	--energy-files energy.npy \
	--force-files qm_force.npy \
	--embed-files qm_type.npy \
	--ext-charge-coord-files mm_coord.npy \
	--ext-charge-files mm_charge.npy \
	--ext-esp-files mm_esp.npy \
	--ext-esp-grad-files mm_esp_grad.npy \
	--wandb-use True --wandb-name tensornet-cas9 --wandb-project cas9

