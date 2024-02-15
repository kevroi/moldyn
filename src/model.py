import functools
import os
import urllib.request
import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import Callable

class MessagePassingModel(nn.Module):
  """A message-passing model for calculating energies and forces of molecules.

    Args:
        features (int): The number of features to use for atomic embeddings. Defaults to 32.
        max_degree (int): The maximum degree of the radial basis functions. Defaults to 2.
        num_iterations (int): The number of message-passing iterations. Defaults to 3.
        num_basis_functions (int): The number of basis functions to use for expansion. Defaults to 8.
        cutoff (float): The cutoff distance for the radial basis functions. Defaults to 5.0.
        max_atomic_number (int): The maximum atomic number in the dataset. Defaults to 118.
        radial_basis_fn (Callable): The radial basis function to use. Defaults to e3x.nn.reciprocal_bernstein.

    Attributes:
        features (int): The number of features to use for atomic embeddings.
        max_degree (int): The maximum degree of the radial basis functions.
        num_iterations (int): The number of message-passing iterations.
        num_basis_functions (int): The number of basis functions to use for expansion.
        cutoff (float): The cutoff distance for the radial basis functions.
        max_atomic_number (int): The maximum atomic number in the dataset.
        radial_basis_fn (Callable): The radial basis function to use.

    Methods:
        energy(atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size):
            Calculate the energy and forces of the molecules.

        __call__(atomic_numbers, positions, dst_idx, src_idx, batch_segments=None, batch_size=None):
            Calculate the energy and forces of the molecules using the `energy` method.
  """
  features: int = 32
  max_degree: int = 2
  num_iterations: int = 3
  num_basis_functions: int = 8
  cutoff: float = 5.0
  max_atomic_number: int = 118
  radial_basis_fn: Callable = e3x.nn.reciprocal_bernstein


  def energy(self, atomic_numbers, positions, dst_idx, src_idx,
             batch_segments, batch_size):
    """Calculate the forces and energy of the molecules.

      Args:
          atomic_numbers (Array): The atomic numbers of the atoms in the molecules.
          positions (Array): The positions of the atoms in the molecules.
          dst_idx (Array): The destination indices for the pairwise interactions.
          src_idx (Array): The source indices for the pairwise interactions.
          batch_segments (Array): The segment ids for the batch.
          batch_size (int): The size of the batch.

      Returns:
          Tuple: A tuple containing the total energy and the forces on each atom.
    """
    # Calculate displacement vectors.
    positions_dst = e3x.ops.gather_dst(positions, dst_idx=dst_idx)
    positions_src = e3x.ops.gather_src(positions, src_idx=src_idx)
    displacements = positions_src - positions_dst  # Shape (num_pairs, 3).

    # Expand displacement vectors in basis functions.
    basis = e3x.nn.basis(  # Shape (num_pairs, 1, (max_degree+1)**2, num_basis_functions).
      displacements,
      num=self.num_basis_functions,
      max_degree=self.max_degree,
      radial_fn=self.radial_basis_fn,
      cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff)
    )

    # Embed atomic numbers in feature space, x has shape (num_atoms, 1, 1, features).
    x = e3x.nn.Embed(num_embeddings=self.max_atomic_number+1, features=self.features)(atomic_numbers)

    # Perform iterations (message-passing + atom-wise refinement).
    for i in range(self.num_iterations):
      # Message-pass.
      if i == self.num_iterations-1:  # Final iteration.
        # Since we will only use scalar features after the final message-pass, we do not want to produce non-scalar
        # features for efficiency reasons.
        y = e3x.nn.MessagePass(max_degree=0, include_pseudotensors=False)(x, basis, dst_idx=dst_idx, src_idx=src_idx)
        # After the final message pass, we can safely throw away all non-scalar features.
        x = e3x.nn.change_max_degree_or_type(x, max_degree=0, include_pseudotensors=False)
      else:
        # In intermediate iterations, the message-pass should consider all possible coupling paths.
        y = e3x.nn.MessagePass()(x, basis, dst_idx=dst_idx, src_idx=src_idx)
      y = e3x.nn.add(x, y)

      # Atom-wise refinement MLP.
      y = e3x.nn.Dense(self.features)(y)
      y = e3x.nn.silu(y)
      y = e3x.nn.Dense(self.features, kernel_init=jax.nn.initializers.zeros)(y)

      # Residual connection.
      x = e3x.nn.add(x, y)

    # Predict atomic energies with an ordinary dense layer.
    element_bias = self.param('element_bias', lambda rng, shape: jnp.zeros(shape), (self.max_atomic_number+1))
    atomic_energies = nn.Dense(1, use_bias=False, kernel_init=jax.nn.initializers.zeros)(x)  # (..., Natoms, 1, 1, 1)
    atomic_energies = jnp.squeeze(atomic_energies, axis=(-1, -2, -3))  # Squeeze last 3 dimensions.
    atomic_energies += element_bias[atomic_numbers]

    energy = jax.ops.segment_sum(atomic_energies, segment_ids=batch_segments, num_segments=batch_size)

    return -jnp.sum(energy), energy  # Forces are the negative gradient, hence the minus sign.


  @nn.compact
  def __call__(self, atomic_numbers, positions, dst_idx, src_idx, batch_segments=None, batch_size=None):
    """Calculate the energy and forces of the molecules using the `energy` method.

      Args:
          atomic_numbers (Array): The atomic numbers of the atoms in the molecules.
          positions (Array): The positions of the atoms in the molecules.
          dst_idx (Array): The destination indices for the pairwise interactions.
          src_idx (Array): The source indices for the pairwise interactions.
          batch_segments (Array): The segment ids for the batch.
          batch_size (int): The size of the batch.

      Returns:
          Tuple: A tuple containing the total energy and the forces on each atom.
    """
    if batch_segments is None:
      batch_segments = jnp.zeros_like(atomic_numbers)
      batch_size = 1

    energy_and_forces = jax.value_and_grad(self.energy, argnums=1, has_aux=True)
    (_, energy), forces = energy_and_forces(atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size)

    return energy, forces