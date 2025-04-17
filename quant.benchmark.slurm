#!/bin/bash
#SBATCH -A cmsc828-class
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH -t 00:30:00
#SBATCH --mem=16G
#SBATCH --job-name=llama_quant
#SBATCH --output=quant_output.txt
#SBATCH --error=quant_error.txt
#SBATCH --cpus-per-task=4

module load cuda/11.7
source activate llama_quant
export TOKENIZERS_PARALLELISM=false
python3 quantization_exp.py

