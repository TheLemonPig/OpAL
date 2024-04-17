#!/bin/sh

#SBATCH -J OpAL-Star-wTests
#SBATCH --account=carney-brainstorm-condo
#SBATCH --time=2:30:00
#SBATCH --array=0-99
#SBATCH --mem=16GB
#SBATCH -n 3
#SBATCH -N 1
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

# Print key runtime properties for records
echo Master process running on `hostname`
echo Directory is `pwd`
echo Starting execution at `date`
echo Current PATH is $PATH

module load graphviz/2.40.1
module load python/3.9.0
module load git/2.29.2
source ~/OpAL/venv/bin/activate
cd /users/jhewson/OpAL/

# Run job
python main_slurm.py --slurm_id $SLURM_ARRAY_TASK_ID