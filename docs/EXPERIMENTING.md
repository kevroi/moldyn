This repo is designed for flexibility in training the model. The following command line argumets can be modified to allow the experiemnter to investigate the suitability and robustness of this e3x model for different molecules, learning rates, e3x basis functions and so on.

For example, to investigate how well our current model architecture performs on a different molecule, we can use the `--molecule` argument to specify the molecule we want to train the model on. The following command trains the model on benzene $\mathrm{C_6H_6}$:

```bash
python train_model.py --molecule "benzene"
```

To change the radial basis function our e3x model uses to represent displacements from the dataset, we can specify the `--rad_bas` argument. The following command trains the model on benzene using a reciprocal Gaussian basis function:

```bash
python train_model.py --molecule "benzene" --rad_bas "rec_gauss"
```

More generally, our repo allows for large scale sweeps over various hyperparameters when using a compute cluster with a SLURM scheduler. To submit single run jobs to a SLURM scheduler, use `job-scripts/single.sh`. For big hyperparameters sweeps (i.e. grid searches), specify the hyper paremerts of interest in `job-scripts/parent.sh` and submit the parent job to the SLURM scheduler.
```bash
sbatch job-scripts/parent.sh
```
The parent job will then submit the actual jobs to the scheduler, modifying the training hyperparameters with each job. The actual job is defined in `job-scripts/child.sh`.

# Full list of command line arguments
The following command line arguments can be used to modify the training process. The default values are shown in square brackets.

- `--molecule`: [`ethanol`] The molecule to train the model on. The molecule must be present in the MD17 or MD22 datasets.
    - `aspirin`
    - `benzene`
    - `ethanol`
    - `malonaldehyde`
    - `naphthalene`
    - `salicylic_acid`
    - `toluene`
    - `uracil`
- `features`: [32] Dimensionality of feature vector to use.
- `--rad_bas`: [`rec_bern`] The radial basis function to use to represent displacements from the dataset. The following radial basis functions are available:
    - `exp_bern`: Exponential Bernstein polynomial
    - `rec_bern`: Reciprocal Bernstein polynomial
    - `exp_cheb`: Exponential Chebyshev polynomial
    - `rec_cheb`: Reciprocal Chebyshev polynomial
    - `exp_gauss`: Exponential Gaussian basis
    - `rec_gauss`: Reciprocal Gaussian basis
- `--n_epochs` (100): The number of epochs to train the model for.
- `--max_degree`: [2] The maximum degree of the polynomial to use in the radial basis function.
- `--num_basis_functions`: [16] The number of basis functions to use in the radial basis function.
- `--cutoff`: [5.0] The cutoff distance to use in the radial basis function. The function is 0 beyond this distance.
- `num_iterations`: [3] The number of iterations to use in the optimizer.
- `forces_weight`: [1.0] The weight to use for the forces loss.
- `--learning_rate`: [0.001] The learning rate to use in the optimizer.
- `--num_epochs`: [100] The number of epochs to train the model for.
- `--batch_size`: [32] The batch size to use in the optimizer.
- `num_train`: [1000] The number of training examples to use. Be wary that your dataset may not contain this many examples.
- `num_val`: [100] The number of validation examples to use.
- `std`: [`sub_mean`]: The standardization to use on the dataset. The following standardizations are available:
    - `sub_mean`: Subtract the mean of the dataset
    - `z_score`: Subtract the mean and divide by the standard deviation of the dataset
    - `min_max`: Subtract the minimum and divide by the range of the dataset
    - `none`: Do not standardize the dataset during pre-processing
- `--use_wandb`: [False] Whether to use `wandb` to log the training process.
