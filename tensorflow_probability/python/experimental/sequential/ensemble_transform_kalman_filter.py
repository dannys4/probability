# Copyright 2020 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Utilities for Ensemble Kalman Filtering."""

import collections
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import distributions
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import distribution_util

__all__ = [
    'ensemble_transform_kalman_filter_update',
]

def _linop_covariance(dist):
  """LinearOperator backing Cov(dist), without unnecessary broadcasting."""
  # This helps, even if we immediately call .to_dense(). Why?
  # Simply calling dist.covariance() would broadcast up to the full batch shape.
  # Instead, we want the shape to be that of the linear operator only.
  # This (i) saves memory and (ii) allows operations done with this operator
  # to be more efficient.
  if hasattr(dist, 'cov_operator'):
    cov = dist.cov_operator
  else:
    cov = dist.scale.matmul(dist.scale.H)
  # TODO(b/132466537) composition doesn't preserve SPD so we have to hard-set.
  cov._is_positive_definite = True  # pylint: disable=protected-access
  cov._is_self_adjoint = True  # pylint: disable=protected-access
  return cov

def _mean_center(x):
  x = tf.convert_to_tensor(x, name='x')
  mu = tf.reduce_mean(x, axis=0, name='mu')
  return mu, x - mu

def _outer_product(X,Y):
  # X (k, ..., m)
  # Y (k, ..., m)
  # Want to do outer product so each inner dimension holds a kxk matrix
  X_T = distribution_util.rotate_transpose(X, -1)
  Y_ref = tf.linalg.matrix_transpose(distribution_util.rotate_transpose(Y,-1))
  return Y_ref @ X_T

def _general_matvec(A,x):
  # A (k, ..., m)
  # x (..., m)
  # Want to do matvec with A^T in (..., k, m) and get something in (k, ...)
  A_T = distribution_util.rotate_transpose(A,-1)
  A_Tx = tf.linalg.matvec(A_T,x,transpose_a=True)
  return distribution_util.rotate_transpose(A_Tx,1)

def ensemble_transform_kalman_filter_update(
    state,
    observation,
    observation_fn,
    damping=1.,
    seed=None,
    name=None):
  """Ensemble Transform Kalman Filter Update.
  Args:
    state: Instance of `EnsembleKalmanFilterState`.
    observation: `Tensor` representing the observation for this timestep.
    observation_fn: callable returning an instance of
      `tfd.MultivariateNormalLinearOperator` along with an extra information
      to be returned in the `EnsembleKalmanFilterState`.
    damping: Floating-point `Tensor` representing how much to damp the
      update by. Used to mitigate filter divergence. Default value: 1.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    name: Python `str` name for ops created by this method.
      Default value: `None` (i.e., `'ensemble_kalman_filter_update'`).
  Returns:
    next_state: `EnsembleKalmanFilterState` representing particles at next
      timestep, after applying Kalman update equations.
  """

  with tf.name_scope(name or 'ensemble_transform_kalman_filter_update'):
    observation_particles_dist, extra = observation_fn(
        state.step, state.particles, state.extra)

    common_dtype = dtype_util.common_dtype(
        [observation_particles_dist, observation], dtype_hint=tf.float32)

    observation = tf.convert_to_tensor(observation, dtype=common_dtype)
    observation_size_is_static_and_scalar = (observation.shape[-1] == 1)

    if not isinstance(observation_particles_dist,
                      distributions.MultivariateNormalLinearOperator):
      raise ValueError('Expected `observation_fn` to return an instance of '
                       '`MultivariateNormalLinearOperator`')

    observation_particles = observation_particles_dist.mean()
    observation_particles_mean, observation_particles_centered = tf.nest.map_structure(_mean_center, observation_particles)
    predicted_state_mean, predicted_state_centered = tf.nest.map_structure(_mean_center, state.particles)

    num_particles = state.particles.shape[0]

    # We specialize the univariate case.
    # TODO(dannys4): Figure out how to adapt this to ETKF
    if observation_size_is_static_and_scalar:
      scaled_observation_covariance = predicted_state_centered / (num_particles-1)*observation_particles_dist.covariance()
      analysis_covariance = 1/((num_particles-1) + scaled_observation_covariance)
      transform_mean = analysis_covariance*scaled_observation_covariance*(observation-observation_particles_mean)
      centered_transform = tf.sqrt((num_particles-1)*analysis_covariance)
      new_particles = predicted_state_mean + predicted_state_centered*(centered_transform+transform_mean)

    else:
      # TODO(dannys4) What to do with `damping` coefficient for ETKF

      # TODO(dannys4) Split between "cheap" observation cov inversion and not
      
      observation_covariance = _linop_covariance(observation_particles_dist)

      # ((k-1)\Gamma)^{-1} G(x)
      scaled_observation_covariance = observation_covariance.solvevec(observation_particles_centered)/(num_particles-1)
      
      # G(x)' ((k-1)\Gamma)^{-1} G(x) + (k-1)I
      """
      G (k, ..., m)
      G' = rotate_transpose(G, -1) (..., m, k)
      Gamma_inv G = matrix_transpose(rotate_transpose(scaled_observation_covariance,-1)) (..., k, m)
      G' Gamma_inv G = (Gamma_inv G) @ G' (..., k, k?)
      """
      # scaled_observation_covariance.H.matvec(observation_particles_centered)
      inv_analysis_covariance = _outer_product(observation_particles_centered, scaled_observation_covariance)
      
      analysis_covariance_eigvals, analysis_covariance_eigvecs = tf.linalg.eigh(inv_analysis_covariance)

      unscaled_transform_mean = _general_matvec(scaled_observation_covariance, observation - observation_particles_mean)
      
      transform_mean = tf.linalg.solve(inv_analysis_covariance, unscaled_transform_mean[...,tf.newaxis])

      centered_transform = tf.linalg.LinearOperatorDiag(tf.sqrt((num_particles - 1)/analysis_covariance_eigvals)).matmul(analysis_covariance_eigvecs)
      
      uncentered_transform = centered_transform + transform_mean

      predicted_state_dev = tf.linalg.matvec(predicted_state_centered, uncentered_transform, adjoint_a=True)

      new_particles = predicted_state_mean + predicted_state_dev

    return tfp.experimental.sequential.EnsembleKalmanFilterState(step=state.step + 1, particles=new_particles, extra=extra)