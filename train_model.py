import functools
import os
import shutil
import urllib.request
import e3x
import flax.linen as nn
from flax.training import orbax_utils
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
from argparse import ArgumentParser
from src.model import MessagePassingModel
from src.training import train_model
from src.utils import prepare_datasets, MOLECULE_CONFIG
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # These appear at each epoch and are not useful for the user.


def main(args):
  # Download the dataset
  save_dir = MOLECULE_CONFIG[args.molecule]["save_dir"]
  filename = MOLECULE_CONFIG[args.molecule]["filename"]
  file_path = "data/"+filename
  if not os.path.exists(file_path):
    print(f"Downloading {file_path} (this may take a while)...")
    urllib.request.urlretrieve(f"http://www.quantum-machine.org/gdml/data/npz/{filename}", file_path)
  else:
    print(f"Dataset {file_path} already exists on your machine. Skipping download.")

  # Remove any existing checkpoints
  if os.path.exists(save_dir):
      shutil.rmtree(save_dir)

  # Create PRNGKeys.
  data_key, train_key = jax.random.split(jax.random.PRNGKey(0), 2)

  # Draw training and validation sets.
  train_data, valid_data, _ = prepare_datasets(file_path, data_key, num_train=args.num_train, num_valid=args.num_valid)

  # Create and train model.
  message_passing_model = MessagePassingModel(
    features=args.features,
    max_degree=args.max_degree,
    num_iterations=args.num_iterations,
    num_basis_functions=args.num_basis_functions,
    cutoff=args.cutoff,
  )
  params = train_model(
    key=train_key,
    model=message_passing_model,
    train_data=train_data,
    valid_data=valid_data,
    num_epochs=args.num_epochs,
    learning_rate=args.learning_rate,
    forces_weight=args.forces_weight,
    batch_size=args.batch_size,
    use_wandb=args.use_wandb,
  )

  # Save the trained parameters to disk.
  orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  save_args = orbax_utils.save_args_from_target(params) # Recommended for performance speedups. It bundles smaller arrays in the pytree to a single large file instead of multiple smaller files.
  orbax_checkpointer.save(save_dir, params, save_args=save_args)

  # Load the trained model parameters from disk.
  reloaded_params = orbax_checkpointer.restore(save_dir)

  # check if we successfully reloaded the params
  assert jax.tree_util.tree_all(jax.tree_map(lambda x, y: (x == y).all(), params, reloaded_params))


if __name__=='__main__':
  parser = ArgumentParser()

  parser.add_argument(
        '--molecule', 
        type=str, 
        default= 'ethanol',
        help='Molecule to analyse'
    )
  parser.add_argument(
        '--features', 
        type=int, 
        default=32, 
        help='Dim of feature vector'
    )
  parser.add_argument(
        '--max_degree', 
        type=int, 
        default=2, 
        help='Max degree of polynomial'
    )
  parser.add_argument(
        '--num_iterations', 
        type=int, 
        default=3, 
        help='Number of iterations'
    )
  parser.add_argument(
        '--num_basis_functions', 
        type=int, 
        default=16, 
        help='Number of basis functions'
    )
  parser.add_argument(
        '--cutoff', 
        type=float, 
        default=5.0, 
        help='Cutoff distance'
    )
  parser.add_argument(
        '--num_train', 
        type=int, 
        default=900, 
        help='Number of training samples'
    )
  parser.add_argument(
        '--num_valid', 
        type=int, 
        default=100, 
        help='Number of validation samples'
    )
  parser.add_argument(
        '--num_epochs', 
        type=int, 
        default=100, 
        help='Number of epochs'
    )
  parser.add_argument(
        '--learning_rate', 
        type=float, 
        default=0.01, 
        help='Learning rate'
    )
  parser.add_argument(
        '--forces_weight', 
        type=float, 
        default=1.0, 
        help='Forces weight'
    )
  parser.add_argument(
        '--batch_size', 
        type=int, 
        default=10, 
        help='Batch size'
    )
  parser.add_argument(
        '--use_wandb', 
        action='store_true', # set to false if we do not pass this argument
        help='Raise the flag to use wandb.'
    )
  args = parser.parse_args()
  main(args)