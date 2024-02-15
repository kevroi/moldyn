# Predicting Molecular Dynamics

This package contains code to train a neural network to predict molecular dynamics. The neural network is trained on a dataset of molecular dynamics trajectories.

## Setup
The current version of the package is not registered yet. To install the package, clone the repository and install the package using the following command:
```bash
pip install requirements.txt
```

## Usage
To train a model, use the `train_model.py` script. The package has been designed to work with the [MD17 dataset](http://www.sgdml.org/#datasets) of molecular trajectories.
```bash
python train_model.py --molecule "ethanol"
```

## Results



## Extra Features

### Logging with `wandb`
To log the training process with `wandb`, use the `--use_wandb` flag. This will log the training process to the `wandb` dashboard. To use this feature, you must have a `wandb` account and be logged in.

## Scheduling Jobs on a SLURM system
To submit single run jobs to a SLURM scheduler, use `job-scripts/single.sh` as follows:
```bash
sbatch job-scripts/single.sh
```
This would train a model with the default command line arguments. To modify the command line arguments, specofy them in the `job-scripts/single.sh` file.

For larger jobs, such as hyperparameter sweeps, use `job-scripts/parent.sh` as follows:
```bash
sbatch job-scripts/parent.sh
```
This will submit a parent job to the SLURM scheduler, which will then submit the actual jobs to the scheduler, modifying the training hyperparameters with each job. The actual job is defined in `job-scripts/child.sh`.

## Notes
The initial model architecture for this package was based on the original e3x [documentation](https://e3x.readthedocs.io/stable/examples/md17_ethanol.html).
