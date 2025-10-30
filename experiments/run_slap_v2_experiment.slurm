#!/bin/bash
#SBATCH --job-name=slap_v2_experiment  # Job name
#SBATCH --time=8:00:00  # Maximum runtime (hh:mm:ss) - longer for training
# #SBATCH --partition=pli
#SBATCH --nodes=1  # Number of nodes
#SBATCH --ntasks=1  # Number of tasks
#SBATCH --cpus-per-task=8  # More CPU cores for parallel training
#SBATCH --mem=64G  # More memory for larger graphs and training
#SBATCH --gres=gpu:1  # Request one GPU


BASE_PATH=${1:-/home/jz4267/shortcut-learning-531}
OUTPUT_PATH=${2:-/scratch/gpfs/jz4267/}

#SBATCH --output=${OUTPUT_PATH}/slap_v2_experiment_%j.out  # Output file with job ID
#SBATCH --error=${OUTPUT_PATH}/slap_v2_experiment_%j.err  # Error file with job ID

# Load necessary modules
module load intel-mkl/2024.2  # Load Intel MKL for numpy

# Set library paths for IKFast (if using obstacle_tower)
export LAPACK_DIR="/usr/lib64"
export LIBGFORTRAN_DIR="/usr/lib64"
export BLAS_DIR="/usr/lib64"

# Activate virtual environment (created in scratch, can be used from anywhere)
module load anaconda3/2024.10
conda activate slap

# Navigate to the experiment directory
cd ${BASE_PATH}/experiments

# Run the SLAP V2 experiment with Hydra
# Override defaults to use slap_v2 approach with multi_rl_ppo_v2 policy
# Increase collection and training for better performance
python run_single_experiment.py \
    approach.approach_type=slap_v2 \
    policy.policy_type=multi_rl_ppo_v2 \
    system=obstacle2d \
    collection.states_per_node=100 \
    training.runs_per_shortcut=1000 \
    training.max_env_steps=50 \
    training.training_record_interval=10 \
    evaluation.num_episodes=100

# Note: Output files are already in /scratch/gpfs/TSILVER/de7281/shortcut_learning/
# Hydra creates its own output directory structure in experiments/outputs/
echo "Experiment complete! Output files are in /scratch/gpfs/TSILVER/de7281/shortcut_learning/"
echo "Hydra outputs are in: $(ls -td experiments/outputs/20*/*/ 2>/dev/null | head -1)"
