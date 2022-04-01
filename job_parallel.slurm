#!/bin/bash
#SBATCH --job-name=model_train    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --mem=4G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --array=0-9
#SBATCH --output=python_array_job_slurm_%A_%a.out
#SBATCH --time=06:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send mail when job begins
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=smalani@princeton.edu

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

module purge
module load anaconda3/2020.11
conda activate torch-env

python train_model_filename.py Case4/model_run_${SLURM_ARRAY_TASK_ID}
# cd URPModel
# python make_plots.py
