import functools
import os
import urllib.request
import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from src.utils import prepare_batches, mean_squared_loss, mean_absolute_error
import wandb


@functools.partial(jax.jit, static_argnames=('model_apply', 'optimizer_update', 'batch_size'))
def train_step(model_apply, optimizer_update, batch, batch_size, forces_weight, opt_state, params):
  """Perform a single training step.

    Args:
        model_apply (callable): The model function to apply.
        optimizer_update (callable): The optimizer update function.
        batch (dict): A dictionary containing the batch data.
        batch_size (int): The batch size.
        forces_weight (float): The weight of the forces in the loss function.
        opt_state (optax.OptState): The optimizer state.
        params (dict): The model parameters.

    Returns:
        Tuple: A tuple containing the updated model parameters, optimizer state, loss, energy MAE, and forces MAE.
  """
  def loss_fn(params):
    energy, forces = model_apply(
      params,
      atomic_numbers=batch['atomic_numbers'],
      positions=batch['positions'],
      dst_idx=batch['dst_idx'],
      src_idx=batch['src_idx'],
      batch_segments=batch['batch_segments'],
      batch_size=batch_size
    )
    loss = mean_squared_loss(
      energy_prediction=energy,
      energy_target=batch['energy'],
      forces_prediction=forces,
      forces_target=batch['forces'],
      forces_weight=forces_weight
    )
    return loss, (energy, forces)
  (loss, (energy, forces)), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
  updates, opt_state = optimizer_update(grad, opt_state, params)
  params = optax.apply_updates(params, updates)
  energy_mae = mean_absolute_error(energy, batch['energy'])
  forces_mae = mean_absolute_error(forces, batch['forces'])
  return params, opt_state, loss, energy_mae, forces_mae


@functools.partial(jax.jit, static_argnames=('model_apply', 'batch_size'))
def eval_step(model_apply, batch, batch_size, forces_weight, params):
  """Perform a single evaluation step.

    Args:
        model_apply (callable): The model function to apply.
        batch (dict): A dictionary containing the batch data.
        batch_size (int): The batch size.
        forces_weight (float): The weight of the forces in the loss function.
        params (dict): The model parameters.

    Returns:
        Tuple: A tuple containing the loss, energy MAE, and forces MAE.
  """
  energy, forces = model_apply(
    params,
    atomic_numbers=batch['atomic_numbers'],
    positions=batch['positions'],
    dst_idx=batch['dst_idx'],
    src_idx=batch['src_idx'],
    batch_segments=batch['batch_segments'],
    batch_size=batch_size
  )
  loss = mean_squared_loss(
    energy_prediction=energy,
    energy_target=batch['energy'],
    forces_prediction=forces,
    forces_target=batch['forces'],
    forces_weight=forces_weight
  )
  energy_mae = mean_absolute_error(energy, batch['energy'])
  forces_mae = mean_absolute_error(forces, batch['forces'])
  return loss, energy_mae, forces_mae


def train_model(cl_args, key, model, train_data, valid_data, num_epochs, learning_rate,
                forces_weight, batch_size, use_wandb=False):
  """Train the model for predicitng molecule's force field and energy.

    Args:
        cl_args: Command-line arguments.
        key (jax.random.PRNGKey): The random key.
        model (nn.Module): The model to train.
        train_data (dict): The training data.
        valid_data (dict): The validation data.
        num_epochs (int): The number of epochs to train for.
        learning_rate (float): The learning rate.
        forces_weight (float): The weight of the forces in the loss function.
        batch_size (int): The batch size.
        use_wandb (bool, optional): Whether to use wandb for logging. Defaults to False.

    Returns:
        dict: The trained model parameters.
  """

  # Initialize model parameters and optimizer state.
  key, init_key = jax.random.split(key)
  optimizer = optax.adam(learning_rate)
  dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(len(train_data['atomic_numbers']))
  params = model.init(init_key,
    atomic_numbers=train_data['atomic_numbers'],
    positions=train_data['positions'][0],
    dst_idx=dst_idx,
    src_idx=src_idx,
  )
  opt_state = optimizer.init(params)

  # Setup logger if using wandb
  if use_wandb:
    config = dict()
    for k, v in vars(cl_args).items():
        if v is not None:
            config[k] = v
    run = wandb.init(
        project="Predicting Molecular Dynamics", 
        config=config,
        save_code=True,
        )

  # Batches for the validation set need to be prepared only once.
  key, shuffle_key = jax.random.split(key)
  valid_batches = prepare_batches(shuffle_key, valid_data, batch_size)

  # Train for 'num_epochs' epochs.
  for epoch in range(1, num_epochs + 1):
    # Prepare batches.
    key, shuffle_key = jax.random.split(key)
    train_batches = prepare_batches(shuffle_key, train_data, batch_size)

    # Loop over train batches.
    train_loss = 0.0
    train_energy_mae = 0.0
    train_forces_mae = 0.0
    for i, batch in enumerate(train_batches):
      params, opt_state, loss, energy_mae, forces_mae = train_step(
        model_apply=model.apply,
        optimizer_update=optimizer.update,
        batch=batch,
        batch_size=batch_size,
        forces_weight=forces_weight,
        opt_state=opt_state,
        params=params
      )
      train_loss += (loss - train_loss)/(i+1)
      train_energy_mae += (energy_mae - train_energy_mae)/(i+1)
      train_forces_mae += (forces_mae - train_forces_mae)/(i+1)

    # Evaluate on validation set.
    valid_loss = 0.0
    valid_energy_mae = 0.0
    valid_forces_mae = 0.0
    for i, batch in enumerate(valid_batches):
      loss, energy_mae, forces_mae = eval_step(
        model_apply=model.apply,
        batch=batch,
        batch_size=batch_size,
        forces_weight=forces_weight,
        params=params
      )
      valid_loss += (loss - valid_loss)/(i+1)
      valid_energy_mae += (energy_mae - valid_energy_mae)/(i+1)
      valid_forces_mae += (forces_mae - valid_forces_mae)/(i+1)

    # Print progress.
    print(f"epoch: {epoch: 3d}                    train:   valid:")
    print(f"    loss [a.u.]             {train_loss : 8.3f} {valid_loss : 8.3f}")
    print(f"    energy mae [kcal/mol]   {train_energy_mae: 8.3f} {valid_energy_mae: 8.3f}")
    print(f"    forces mae [kcal/mol/Å] {train_forces_mae: 8.3f} {valid_forces_mae: 8.3f}")

    # Log to wandb if using it
    if use_wandb:
      wandb.log({'train_loss': train_loss, 'valid_loss': valid_loss,
                 'train_energy_mae': train_energy_mae, 'valid_energy_mae': valid_energy_mae,
                 'train_forces_mae': train_forces_mae, 'valid_forces_mae': valid_forces_mae,
                 'epoch': epoch})


  # Return final model parameters.
  return params