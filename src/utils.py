import functools
import os
import urllib.request
import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import ase
import ase.calculators.calculator as ase_calc

# MD17 Dataset hyperlinks for each molecule
MOLECULE_CONFIG = {
  'aspirin': {
    'filename': "md17_aspirin.npz",
    'save_dir': "/tmp/Mol_Dynamics/MD17_aspirin",
    },
  'benzene': {
    'filename': "md17_benzene2017.npz",
    'save_dir': "/tmp/Mol_Dynamics/MD17_benzene",
    },
  'ethanol': {
    'filename': "md17_ethanol.npz",
    'save_dir': "/tmp/Mol_Dynamics/MD17_ethanol",
    },
  'malonaldehyde': {
    'filename': "md17_malonaldehyde.npz",
    'save_dir': "/tmp/Mol_Dynamics/MD17_malonaldehyde",
    },
  'naphthalene': {
    'filename': "md17_naphthalene.npz",
    'save_dir': "/tmp/Mol_Dynamics/MD17_naphthalene",
    },
  'salicylic_acid': {
    'filename': "md17_salicylic_acid.npz",
    'save_dir': "/tmp/Mol_Dynamics/MD17_salicylic_acid",
    },
  'toluene': {
    'filename': "md17_toluene.npz",
    'save_dir': "/tmp/Mol_Dynamics/MD17_toluene",
    },
  'uracil': {
    'filename': "md17_uracil.npz",
    'save_dir': "/tmp/Mol_Dynamics/MD17_uracil",
    },
}

RADIAL_BASIS_CONFIG = {
  'exp_bern': e3x.nn.exponential_bernstein,
  'rec_bern': e3x.nn.reciprocal_bernstein,
  'exp_cheb': e3x.nn.exponential_chebyshev,
  'rec_cheb': e3x.nn.reciprocal_chebyshev,
  'exp_gauss': e3x.nn.exponential_gaussian,
  'rec_gauss': e3x.nn.reciprocal_gaussian,
  'sinc': e3x.nn.sinc,
}

def prepare_datasets(filename, key, num_train=900, num_valid=100, standardization='sub_mean'):
  """Prepare training and validation datasets.

    Args:
        filename (str): The filename of the dataset.
        key (jax.random.PRNGKey): The random key.
        num_train (int): The number of training samples. Defaults to 900.
        num_valid (int): The number of validation samples. Defaults to 100.
        standardization (str): The type of energy standardization to apply. Can be 'sub_mean', 'z_score', 'min_max', or None. Defaults to 'sub_mean'.

    Returns:
        tuple: A tuple containing the training and validation datasets.

    Raises:
        RuntimeError: If the dataset does not contain enough entries.
  """
  # Load the dataset.
  dataset = np.load(filename)

  # Make sure that the dataset contains enough entries.
  num_data = len(dataset['E'])
  num_draw = num_train + num_valid
  if num_draw > num_data:
    raise RuntimeError(
      f'datasets only contains {num_data} points, requested num_train={num_train}, num_valid={num_valid}')

  # Randomly draw train and validation sets from dataset.
  choice = np.asarray(jax.random.choice(key, num_data, shape=(num_draw,), replace=False))
  train_choice = choice[:num_train]
  valid_choice = choice[num_train:]

  # standardize the energy
  if standardization == 'sub_mean':
    mean_energy = np.mean(dataset['E'][train_choice])
    train_data = dict(
      energy=jnp.asarray(dataset['E'][train_choice, 0] - mean_energy),
      forces=jnp.asarray(dataset['F'][train_choice]),
      atomic_numbers=jnp.asarray(dataset['z']),
      positions=jnp.asarray(dataset['R'][train_choice]),
    )
    valid_data = dict(
      energy=jnp.asarray(dataset['E'][valid_choice, 0] - mean_energy),
      forces=jnp.asarray(dataset['F'][valid_choice]),
      atomic_numbers=jnp.asarray(dataset['z']),
      positions=jnp.asarray(dataset['R'][valid_choice]),
    )
    print("MEAN_ENERGY: ", mean_energy)
  elif standardization == 'z_score':
    mean_energy = np.mean(dataset['E'][train_choice])
    std_energy = np.std(dataset['E'][train_choice])
    train_data = dict(
      energy=jnp.asarray((dataset['E'][train_choice, 0] - mean_energy) / std_energy),
      forces=jnp.asarray(dataset['F'][train_choice]),
      atomic_numbers=jnp.asarray(dataset['z']),
      positions=jnp.asarray(dataset['R'][train_choice]),
    )
    valid_data = dict(
      energy=jnp.asarray((dataset['E'][valid_choice, 0] - mean_energy) / std_energy),
      forces=jnp.asarray(dataset['F'][valid_choice]),
      atomic_numbers=jnp.asarray(dataset['z']),
      positions=jnp.asarray(dataset['R'][valid_choice]),
    )
    print("MEAN_ENERGY: ", mean_energy)
    print("STD_ENERGY: ", std_energy)
  elif standardization == 'min_max':
    min_energy = np.min(dataset['E'][train_choice])
    max_energy = np.max(dataset['E'][train_choice])
    train_data = dict(
      energy=jnp.asarray((dataset['E'][train_choice, 0] - min_energy) / (max_energy - min_energy)),
      forces=jnp.asarray(dataset['F'][train_choice]),
      atomic_numbers=jnp.asarray(dataset['z']),
      positions=jnp.asarray(dataset['R'][train_choice]),
    )
    valid_data = dict(
      energy=jnp.asarray((dataset['E'][valid_choice, 0] - min_energy) / (max_energy - min_energy)),
      forces=jnp.asarray(dataset['F'][valid_choice]),
      atomic_numbers=jnp.asarray(dataset['z']),
      positions=jnp.asarray(dataset['R'][valid_choice]),
    )
    print("MIN_ENERGY: ", min_energy)
    print("MAX_ENERGY: ", max_energy)
  else:
    train_data = dict(
      energy=jnp.asarray(dataset['E'][train_choice, 0]),
      forces=jnp.asarray(dataset['F'][train_choice]),
      atomic_numbers=jnp.asarray(dataset['z']),
      positions=jnp.asarray(dataset['R'][train_choice]),
    )
    valid_data = dict(
      energy=jnp.asarray(dataset['E'][valid_choice, 0]),
      forces=jnp.asarray(dataset['F'][valid_choice]),
      atomic_numbers=jnp.asarray(dataset['z']),
      positions=jnp.asarray(dataset['R'][valid_choice]),
    )
  return train_data, valid_data


def prepare_batches(key, data, batch_size):
  """Prepare batches for training.

    Args:
        key (jax.random.PRNGKey): The random key.
        data (dict): The data dictionary.
        batch_size (int): The batch size.

    Returns:
        list: A list of batches.
  """
  # Determine the number of training steps per epoch.
  data_size = len(data['energy'])
  steps_per_epoch = data_size//batch_size

  # Draw random permutations for fetching batches from the train data.
  perms = jax.random.permutation(key, data_size)
  perms = perms[:steps_per_epoch * batch_size]  # Skip the last batch (if incomplete).
  perms = perms.reshape((steps_per_epoch, batch_size))

  # Prepare entries that are identical for each batch.
  num_atoms = len(data['atomic_numbers'])
  batch_segments = jnp.repeat(jnp.arange(batch_size), num_atoms)
  atomic_numbers = jnp.tile(data['atomic_numbers'], batch_size)
  offsets = jnp.arange(batch_size) * num_atoms
  dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
  dst_idx = (dst_idx + offsets[:, None]).reshape(-1)
  src_idx = (src_idx + offsets[:, None]).reshape(-1)

  # Assemble and return batches.
  return [
    dict(
        energy=data['energy'][perm],
        forces=data['forces'][perm].reshape(-1, 3),
        atomic_numbers=atomic_numbers,
        positions=data['positions'][perm].reshape(-1, 3),
        dst_idx=dst_idx,
        src_idx=src_idx,
        batch_segments = batch_segments,
    )
    for perm in perms
  ]

## LOSS FUNCTIONS ##
def mean_squared_loss(energy_prediction, energy_target, forces_prediction, forces_target, forces_weight):
  """Calculate the mean squared loss.

    Args:
        energy_prediction: The predicted energy.
        energy_target: The target energy.
        forces_prediction: The predicted forces.
        forces_target: The target forces.
        forces_weight: The weight of the forces in the loss function.

    Returns:
        jnp.ndarray: The mean squared loss.
  """
  energy_loss = jnp.mean(optax.l2_loss(energy_prediction, energy_target))
  forces_loss = jnp.mean(optax.l2_loss(forces_prediction, forces_target))
  return energy_loss + forces_weight * forces_loss

def mean_absolute_error(prediction, target):
  """Calculate the mean absolute error.

    Args:
        prediction: The predicted values.
        target: The target values.

    Returns:
        jnp.ndarray: The mean absolute error.
  """
  return jnp.mean(jnp.abs(prediction - target))

def huber_loss(y_true, y_pred, delta=1.0):
  """Calculate the huber loss.

    Args:
        y_true: The true values.
        y_pred: The predicted values.
        delta: The threshold for the loss function.

    Returns:
        jnp.ndarray: The huber loss.
  """
  residual = jnp.abs(y_true - y_pred)
  return jnp.mean(jnp.where(residual < delta, 0.5 * jnp.square(residual), delta * (residual - 0.5 * delta)))