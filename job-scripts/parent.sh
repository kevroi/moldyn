#!/bin/bash
#SBATCH --job-name=parent_job
#SBATCH --account=rrg-whitem
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-2:55
#SBATCH --mail-type=ALL
#SBATCH --mail-user=roice@ualberta.ca
#SBATCH -o /home/roice/scratch/discovery/logs/%x.out
#SBATCH -e /home/roice/scratch/discovery/logs/%x.err

# These overwrite config.yaml
molecules=("aspirin", "benzene", "ethanol", "toluene")
learning_rates=(0.0001, 0.001, 0.01, 0.1, 1.0)
# batch_sizes=(5, 10, 20, 40, 80)

# Loop over the parameter sweep values
for mol in ${molecules[@]}; do
    for lr in ${learning_rates[@]}; do
        # Submit a job for each parameter combination
        sbatch --job-name="${mol}_${lr}" --export=MOL="$mol",LR="$lr" job-scripts/child.sh
        # sleep for 2 seconds to prevent overloading the scheduler
        srun sleep 2
    done
done
