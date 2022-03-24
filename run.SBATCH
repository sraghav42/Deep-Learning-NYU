#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu
#SBATCH --job-name=torch

module purge

singularity exec --nv --overlay $SCRATCH/singularity/overlay-7.5GB-300K.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python s_project1_model.py"
