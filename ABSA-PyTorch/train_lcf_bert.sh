#!/usr/local_rwth/bin/zsh

#SBATCH --job-name=train_lcf_bert
#SBATCH --output=cluster_logs/lcf_%J.txt
#SBATCH --time=11:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:volta:1

module load cuda
module load cudnn/7.4

source ../tm_env/bin/activate

echo "Starting training."

python3 train.py --model_name lcf_bert --dataset combined

echo "Script completed."