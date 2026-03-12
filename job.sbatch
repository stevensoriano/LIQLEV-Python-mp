#!/bin/bash
#SBATCH --job-name=liqlev-mp
#SBATCH --output=liqlev-mp_%j.out
#SBATCH --error=liqlev-mp_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=04:00:00

echo "=========================================="
echo "LIQLEV-Python-parallel"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "=========================================="

python main.py

echo "Job finished at $(date)"
