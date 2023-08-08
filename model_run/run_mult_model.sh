#!/bin/bash --login
#SBATCH --job-name=lm_temporal_classifcation
#SBATCH --output=/scratch/c.c2075248/wiki_weakly_supervised_classifier/syslog/out_bart_finetune_tempdef.txt.%J
#SBATCH --error=/scratch/c.c2075248/wiki_weakly_supervised_classifier/syslog//err_bart_finetune_tempdef.%J
#SBATCH --tasks-per-node=5
#SBATCH --mem=64g
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu7d
#SBATCH --time=7-00:00:00
#SBATCH -p gpu_v100
#SBATCH --ntasks=5
#SBATCH -A scw1971
#SBATCH --exclusive



module load anaconda/2021.11
conda activate projectenv


echo 'Running experiment...'

python3 /scratch/c.c2075248/wiki_weakly_supervised_classifier/src/LM_multiple_model.py  -i /scratch/c.c2075248/wiki_weakly_supervised_classifier/data -o /scratch/c.c2075248/wiki_weakly_supervised_classifier/output/
echo finished!