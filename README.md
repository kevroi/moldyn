
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

### Commits
This package follows the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) standard. This means that each commit message should be of the form:
```
<type>[optional scope]: <description>
```
Where `type` is one of the following:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools and libraries such as documentation generation
