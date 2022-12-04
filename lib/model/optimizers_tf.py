#!/usr/bin/env python3
""" Custom Optimizers for TensorFlow 2.x/tf.keras """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import sys

import tensorflow as tf

# Ignore linting errors from Tensorflow's thoroughly broken import system
from tensorflow.keras.optimizers import (Adam, Nadam, RMSprop)  # noqa pylint:disable=no-name-in-module,unused-import,import-error
from tensorflow.keras.utils import get_custom_objects  # noqa pylint:disable=no-name-in-module,import-error
#from tensorflow.keras.optimizers.experimental import AdamW # noqa pylint:disable=no-name-in-module,import-error
from tensorflow.keras import backend as K  # noqa pylint:disable=no-name-in-module,import-error

import tensorflow_probability as tfp

class AdaBelief(tf.keras.optimizers.Optimizer):
    """ Implementation of the AdaBelief Optimizer

    Inherits from: tf.keras.optimizers.Optimizer.

    AdaBelief Optimizer is not a placement of the heuristic warmup, the settings should be kept if
    warmup has already been employed and tuned in the baseline method. You can enable warmup by
    setting `total_steps` and `warmup_proportion` (see examples)

    Lookahead (see references) can be integrated with AdaBelief Optimizer, which is announced by
    Less Wright and the new combined optimizer can also be called "Ranger". The mechanism can be
    enabled by using the lookahead wrapper. (See examples)

    Parameters
    ----------
    learning_rate: `Tensor`, float or :class: `tf.keras.optimizers.schedules.LearningRateSchedule`
        The learning rate.
    beta_1: float
        The exponential decay rate for the 1st moment estimates.
    beta_2: float
        The exponential decay rate for the 2nd moment estimates.
    epsilon: float
        A small constant for numerical stability.
    weight_decay: `Tensor`, float or :class: `tf.keras.optimizers.schedules.LearningRateSchedule`
        Weight decay for each parameter.
    rectify: bool
        Whether to enable rectification as in RectifiedAdam
    amsgrad: bool
        Whether to apply AMSGrad variant of this algorithm from the paper "On the Convergence
        of Adam and beyond".
    sma_threshold. float
        The threshold for simple mean average.
    total_steps: int
        Total number of training steps. Enable warmup by setting a positive value.
    warmup_proportion: float
        The proportion of increasing steps.
    min_lr: float
        Minimum learning rate after warmup.
    name: str, optional
        Name for the operations created when applying gradients. Default: ``"AdaBeliefOptimizer"``.
    **kwargs: dict
        Standard Keras Optimizer keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip gradients by value,
        `decay` is included for backward compatibility to allow time inverse decay of learning
        rate. `lr` is included for backward compatibility, recommended to use `learning_rate`
        instead.

    Examples
    --------
    >>> from adabelief_tf import AdaBelief
    >>> opt = AdaBelief(lr=1e-3)

    Example of serialization:

    >>> optimizer = AdaBelief(learning_rate=lr_scheduler, weight_decay=wd_scheduler)
    >>> config = tf.keras.optimizers.serialize(optimizer)
    >>> new_optimizer = tf.keras.optimizers.deserialize(config,
    ...                                                 custom_objects=dict(AdaBelief=AdaBelief))

    Example of warm up:

    >>> opt = AdaBelief(lr=1e-3, total_steps=10000, warmup_proportion=0.1, min_lr=1e-5)

    In the above example, the learning rate will increase linearly from 0 to `lr` in 1000 steps,
    then decrease linearly from `lr` to `min_lr` in 9000 steps.

    Example of enabling Lookahead:

    >>> adabelief = AdaBelief()
    >>> ranger = tfa.optimizers.Lookahead(adabelief, sync_period=6, slow_step_size=0.5)

    Notes
    -----
    `amsgrad` is not described in the original paper. Use it with caution.

    References
    ----------
    Juntang Zhuang et al. - AdaBelief Optimizer: Adapting stepsizes by the belief in observed
    gradients - https://arxiv.org/abs/2010.07468.

    Original implementation - https://github.com/juntang-zhuang/Adabelief-Optimizer

    Michael R. Zhang et.al - Lookahead Optimizer: k steps forward, 1 step back -
    https://arxiv.org/abs/1907.08610v1

    Adapted from https://github.com/juntang-zhuang/Adabelief-Optimizer

    BSD 2-Clause License

    Copyright (c) 2021, Juntang Zhuang
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-14,
                 weight_decay=0.0, rectify=True, amsgrad=False, sma_threshold=5.0, total_steps=0,
                 warmup_proportion=0.1, min_lr=0.0, name="AdaBeliefOptimizer", **kwargs):
        # pylint:disable=too-many-arguments
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("weight_decay", weight_decay)
        self._set_hyper("sma_threshold", sma_threshold)
        self._set_hyper("total_steps", int(total_steps))
        self._set_hyper("warmup_proportion", warmup_proportion)
        self._set_hyper("min_lr", min_lr)
        self.epsilon = epsilon or tf.keras.backend.epsilon()
        self.amsgrad = amsgrad
        self.rectify = rectify
        self._has_weight_decay = weight_decay != 0.0
        self._initial_total_steps = total_steps

    def _create_slots(self, var_list):
        """ Create slots for the first and second moments

        Parameters
        ----------
        var_list: list
            List of tensorflow variables to create slots for
        """
        for var in var_list:
            self.add_slot(var, "m")
            self.add_slot(var, "v")
            if self.amsgrad:
                self.add_slot(var, "vhat")

    def set_weights(self, weights):
        """ Set the weights of the optimizer.

        The weights of an optimizer are its state (IE, variables). This function takes the weight
        values associated with this optimizer as a list of Numpy arrays. The first value is always
        the iterations count of the optimizer, followed by the optimizers state variables in the
        order they are created. The passed values are used to set the new state of the optimizer.

        Parameters
        ----------
        weights: list
            weight values as a list of numpy arrays.
        """
        params = self.weights
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[: len(params)]
        super().set_weights(weights)

    def _decayed_wd(self, var_dtype):
        """ Set the weight decay

        Parameters
        ----------
        var_dtype: str
            The data type to to set up weight decay for

        Returns
        -------
        Tensor
            The weight decay variable
        """
        wd_t = self._get_hyper("weight_decay", var_dtype)
        if isinstance(wd_t, tf.keras.optimizers.schedules.LearningRateSchedule):
            wd_t = tf.cast(wd_t(self.iterations), var_dtype)
        return wd_t

    def _resource_apply_dense(self, grad, handle, apply_state=None):
        # pylint:disable=too-many-locals,unused-argument
        """ Add ops to apply dense gradients to the variable handle.

        Parameters
        ----------
        grad: Tensor
            A tensor representing the gradient.
        handle: Tensor
            a Tensor of dtype resource which points to the variable to be updated.
        apply_state: dict
            A dict which is used across multiple apply calls.

        Returns
        -------
            An Operation which updates the value of the variable.
        """
        var_dtype = handle.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        wd_t = self._decayed_wd(var_dtype)
        var_m = self.get_slot(handle, "m")
        var_v = self.get_slot(handle, "v")
        beta_1_t = self._get_hyper("beta_1", var_dtype)
        beta_2_t = self._get_hyper("beta_2", var_dtype)
        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_power = tf.math.pow(beta_1_t, local_step)
        beta_2_power = tf.math.pow(beta_2_t, local_step)

        if self._initial_total_steps > 0:
            total_steps = self._get_hyper("total_steps", var_dtype)
            warmup_steps = total_steps * self._get_hyper("warmup_proportion", var_dtype)
            min_lr = self._get_hyper("min_lr", var_dtype)
            decay_steps = tf.maximum(total_steps - warmup_steps, 1)
            decay_rate = (min_lr - lr_t) / decay_steps
            lr_t = tf.where(local_step <= warmup_steps,
                            lr_t * (local_step / warmup_steps),
                            lr_t + decay_rate * tf.minimum(local_step - warmup_steps, decay_steps))

        m_t = var_m.assign(beta_1_t * var_m + (1.0 - beta_1_t) * grad,
                           use_locking=self._use_locking)
        m_corr_t = m_t / (1.0 - beta_1_power)

        v_t = var_v.assign(
            beta_2_t * var_v + (1.0 - beta_2_t) * tf.math.square(grad - m_t) + epsilon_t,
            use_locking=self._use_locking)

        if self.amsgrad:
            vhat = self.get_slot(handle, "vhat")
            vhat_t = vhat.assign(tf.maximum(vhat, v_t), use_locking=self._use_locking)
            v_corr_t = tf.math.sqrt(vhat_t / (1.0 - beta_2_power))
        else:
            vhat_t = None
            v_corr_t = tf.math.sqrt(v_t / (1.0 - beta_2_power))

        if self.rectify:
            sma_inf = 2.0 / (1.0 - beta_2_t) - 1.0
            sma_t = sma_inf - 2.0 * local_step * beta_2_power / (1.0 - beta_2_power)
            r_t = tf.math.sqrt((sma_t - 4.0) / (sma_inf - 4.0) *
                               (sma_t - 2.0) / (sma_inf - 2.0) *
                               sma_inf / sma_t)
            sma_threshold = self._get_hyper("sma_threshold", var_dtype)
            var_t = tf.where(sma_t >= sma_threshold,
                             r_t * m_corr_t / (v_corr_t + epsilon_t),
                             m_corr_t)
        else:
            var_t = m_corr_t / (v_corr_t + epsilon_t)

        if self._has_weight_decay:
            var_t += wd_t * handle

        var_update = handle.assign_sub(lr_t * var_t, use_locking=self._use_locking)
        updates = [var_update, m_t, v_t]

        if self.amsgrad:
            updates.append(vhat_t)
        return tf.group(*updates)

    def _resource_apply_sparse(self, grad, handle, indices, apply_state=None):
        # pylint:disable=too-many-locals, unused-argument
        """ Add ops to apply sparse gradients to the variable handle.

        Similar to _apply_sparse, the indices argument to this method has been de-duplicated.
        Optimizers which deal correctly with non-unique indices may instead override
        :func:`_resource_apply_sparse_duplicate_indices` to avoid this overhead.

        Parameters
        ----------
        grad: Tensor
            a Tensor representing the gradient for the affected indices.
        handle: Tensor
            a Tensor of dtype resource which points to the variable to be updated.
        indices: Tensor
            a Tensor of integral type representing the indices for which the gradient is nonzero.
            Indices are unique.
        apply_state: dict
            A dict which is used across multiple apply calls.

        Returns
        -------
            An Operation which updates the value of the variable.
        """
        var_dtype = handle.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        wd_t = self._decayed_wd(var_dtype)
        beta_1_t = self._get_hyper("beta_1", var_dtype)
        beta_2_t = self._get_hyper("beta_2", var_dtype)
        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_power = tf.math.pow(beta_1_t, local_step)
        beta_2_power = tf.math.pow(beta_2_t, local_step)

        if self._initial_total_steps > 0:
            total_steps = self._get_hyper("total_steps", var_dtype)
            warmup_steps = total_steps * self._get_hyper("warmup_proportion", var_dtype)
            min_lr = self._get_hyper("min_lr", var_dtype)
            decay_steps = tf.maximum(total_steps - warmup_steps, 1)
            decay_rate = (min_lr - lr_t) / decay_steps
            lr_t = tf.where(local_step <= warmup_steps,
                            lr_t * (local_step / warmup_steps),
                            lr_t + decay_rate * tf.minimum(local_step - warmup_steps, decay_steps))

        var_m = self.get_slot(handle, "m")
        m_scaled_g_values = grad * (1 - beta_1_t)
        m_t = var_m.assign(var_m * beta_1_t, use_locking=self._use_locking)
        m_t = self._resource_scatter_add(var_m, indices, m_scaled_g_values)
        m_corr_t = m_t / (1.0 - beta_1_power)

        var_v = self.get_slot(handle, "v")
        m_t_indices = tf.gather(m_t, indices)  # pylint:disable=no-value-for-parameter
        v_scaled_g_values = tf.math.square(grad - m_t_indices) * (1 - beta_2_t)
        v_t = var_v.assign(var_v * beta_2_t + epsilon_t, use_locking=self._use_locking)
        v_t = self._resource_scatter_add(var_v, indices, v_scaled_g_values)

        if self.amsgrad:
            vhat = self.get_slot(handle, "vhat")
            vhat_t = vhat.assign(tf.maximum(vhat, v_t), use_locking=self._use_locking)
            v_corr_t = tf.math.sqrt(vhat_t / (1.0 - beta_2_power))
        else:
            vhat_t = None
            v_corr_t = tf.math.sqrt(v_t / (1.0 - beta_2_power))

        if self.rectify:
            sma_inf = 2.0 / (1.0 - beta_2_t) - 1.0
            sma_t = sma_inf - 2.0 * local_step * beta_2_power / (1.0 - beta_2_power)
            r_t = tf.math.sqrt((sma_t - 4.0) / (sma_inf - 4.0) *
                               (sma_t - 2.0) / (sma_inf - 2.0) *
                               sma_inf / sma_t)
            sma_threshold = self._get_hyper("sma_threshold", var_dtype)
            var_t = tf.where(sma_t >= sma_threshold,
                             r_t * m_corr_t / (v_corr_t + epsilon_t),
                             m_corr_t)
        else:
            var_t = m_corr_t / (v_corr_t + epsilon_t)

        if self._has_weight_decay:
            var_t += wd_t * handle

        var_update = self._resource_scatter_add(handle,
                                                indices,
                                                tf.gather(  # pylint:disable=no-value-for-parameter
                                                    tf.math.negative(lr_t) * var_t,
                                                    indices))

        updates = [var_update, m_t, v_t]
        if self.amsgrad:
            updates.append(vhat_t)
        return tf.group(*updates)

    def get_config(self):
        """ Returns the config of the optimizer.

        An optimizer config is a Python dictionary (serializable) containing the configuration of
        an optimizer. The same optimizer can be re-instantiated later (without any saved state)
        from this configuration.

        Returns
        -------
        dict
            The optimizer configuration.
        """
        config = super().get_config()
        config.update(dict(learning_rate=self._serialize_hyperparameter("learning_rate"),
                           beta_1=self._serialize_hyperparameter("beta_1"),
                           beta_2=self._serialize_hyperparameter("beta_2"),
                           decay=self._serialize_hyperparameter("decay"),
                           weight_decay=self._serialize_hyperparameter("weight_decay"),
                           sma_threshold=self._serialize_hyperparameter("sma_threshold"),
                           epsilon=self.epsilon,
                           amsgrad=self.amsgrad,
                           rectify=self.rectify,
                           total_steps=self._serialize_hyperparameter("total_steps"),
                           warmup_proportion=self._serialize_hyperparameter("warmup_proportion"),
                           min_lr=self._serialize_hyperparameter("min_lr")))
        return config



# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Adam optimizer implementation."""

## TODO: amsgrad, sparse, config
## performance hit (1/2 speed)?

class AdamLRD(Adam):
    def __init__(self,
            lr_dropout=0.3,
            **kwargs):
        super().__init__(**kwargs)
        self.lr_dropout = lr_dropout
        self.distrib = tfp.distributions.Bernoulli(probs=self.lr_dropout, dtype=tf.float32)
    
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                    self._fallback_apply_state(var_device, var_dtype))

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        alpha =  coefficients['lr_t'] * (tf.sqrt(1 - coefficients['beta_2_power']) / (1 - coefficients['beta_1_power'])) * self.distrib.sample(m.shape)
        
        m.assign_add((grad - m) * (1 - coefficients['beta_1_t']))
        v.assign_add((tf.square(grad) - v) * (1 - coefficients['beta_2_t']))
    
        var_update = var.assign_sub((m * alpha) / (tf.sqrt(v) + coefficients['epsilon']))

        return tf.group(*[var_update, m, v])


## TODO: make mixed precision work with experimental

from keras.optimizers.optimizer_experimental import optimizer

class AdamLRD2(optimizer.Optimizer):
    r"""Optimizer that implements the Adam algorithm.

    Adam optimization is a stochastic gradient descent method that is based on
    adaptive estimation of first-order and second-order moments.

    According to
    [Kingma et al., 2014](http://arxiv.org/abs/1412.6980),
    the method is "*computationally
    efficient, has little memory requirement, invariant to diagonal rescaling of
    gradients, and is well suited for problems that are large in terms of
    data/parameters*".

    Args:
    learning_rate: A `tf.Tensor`, floating point value, a schedule that is a
        `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
        that takes no arguments and returns the actual value to use. The
        learning rate. Defaults to 0.001.
    lr_dropout: A float value or a constant float tensor. The parameter used
        for Bernoulli distribution. Defaults to 0.3.
    beta_1: A float value or a constant float tensor, or a callable
        that takes no arguments and returns the actual value to use. The
        exponential decay rate for the 1st moment estimates. Defaults to 0.9.
    beta_2: A float value or a constant float tensor, or a callable
        that takes no arguments and returns the actual value to use. The
        exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
    epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
        1e-7.
    amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm from
        the paper "On the Convergence of Adam and beyond". Defaults to `False`.
    {{base_optimizer_keyword_args}}

    Reference:
    - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
    - [Reddi et al., 2018](
        https://openreview.net/pdf?id=ryQu7f-RZ) for `amsgrad`.

    Notes:

    The default value of 1e-7 for epsilon might not be a good default in
    general. For example, when training an Inception network on ImageNet a
    current good choice is 1.0 or 0.1. Note that since Adam uses the
    formulation just before Section 2.1 of the Kingma and Ba paper rather than
    the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
    hat" in the paper.

    The sparse implementation of this algorithm (used when the gradient is an
    IndexedSlices object, typically because of `tf.gather` or an embedding
    lookup in the forward pass) does apply momentum to variable slices even if
    they were not used in the forward pass (meaning they have a gradient equal
    to zero). Momentum decay (beta1) is also applied to the entire momentum
    accumulator. This means that the sparse behavior is equivalent to the dense
    behavior (in contrast to some momentum implementations which ignore momentum
    unless a variable slice was actually used).
    """

    def __init__(self,
                learning_rate=0.001,
                lr_dropout=0.3,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7,
                amsgrad=False,
                clipnorm=None,
                clipvalue=None,
                global_clipnorm=None,
                use_ema=False,
                ema_momentum=0.99,
                ema_overwrite_frequency=None,
                jit_compile=True,
                name='AdamLRD',
                **kwargs):
        super().__init__(
            name=name,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            jit_compile=jit_compile,
            **kwargs)
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.lr_dropout = lr_dropout
        self.distrib = tfp.distributions.Bernoulli(probs=self.lr_dropout)

    def build(self, var_list):
        """Initialize optimizer variables.

        Adam optimizer has 3 types of variables: momentums, velocities and
        velocity_hat (only set when amsgrad is applied),

        Args:
            var_list: list of model variables to build Adam variables on.
        """
        super().build(var_list)
        if hasattr(self, '_built') and self._built:
            return
        self._built = True
        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name='m'))
            self._velocities.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name='v'))
        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(
                    self.add_variable_from_reference(
                        model_variable=var, variable_name='vhat'))

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        beta_1_power = None
        beta_2_power = None
        lr = tf.cast(self.learning_rate, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        beta_1_power = tf.pow(tf.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = tf.pow(tf.cast(self.beta_2, variable.dtype), local_step)

        var_key = self._var_key(variable)
        m = self._momentums[self._index_dict[var_key]]
        v = self._velocities[self._index_dict[var_key]]

        alpha =  lr * (tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)) * self.distrib.sample(m.shape)

        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            m.assign_add(-m * (1 - self.beta_1))
            m.scatter_add(
                tf.IndexedSlices(gradient.values * (1 - self.beta_1),
                                gradient.indices))
            v.assign_add(-v * (1 - self.beta_2))
            v.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - self.beta_2), gradient.indices))
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
                variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))
        else:
            # Dense gradients.
            m.assign_add((gradient - m) * (1 - self.beta_1))
            v.assign_add((tf.square(gradient) - v) * (1 - self.beta_2))
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))

    def get_config(self):
        config = super().get_config()

        config.update({
            'learning_rate': self._serialize_hyperparameter(self._learning_rate),
            'lr_dropout': self.lr_dropout,
            'beta_1': self.beta_1,
            'beta_2': self.beta_2,
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
        })
        return config


# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================


"""AdamW optimizer implementation."""

import re

class AdamW(tf.keras.optimizers.Optimizer):
    r"""Optimizer that implements the AdamW algorithm.
    AdamW optimization is a stochastic gradient descent method that is based on
    adaptive estimation of first-order and second-order moments with an added
    method to decay weights per the techniques discussed in the paper,
    'Decoupled Weight Decay Regularization' by
    [Loshchilov, Hutter et al., 2019](https://arxiv.org/abs/1711.05101).
    According to
    [Kingma et al., 2014](http://arxiv.org/abs/1412.6980),
    the underying Adam method is "*computationally
    efficient, has little memory requirement, invariant to diagonal rescaling of
    gradients, and is well suited for problems that are large in terms of
    data/parameters*".
    Args:
      learning_rate: A `tf.Tensor`, floating point value, a schedule that is a
        `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
        that takes no arguments and returns the actual value to use. The
        learning rate. Defaults to 0.001.
      weight_decay: A `tf.Tensor`, floating point value. The weight decay.
        Defaults to 0.004.
      beta_1: A float value or a constant float tensor, or a callable
        that takes no arguments and returns the actual value to use. The
        exponential decay rate for the 1st moment estimates. Defaults to 0.9.
      beta_2: A float value or a constant float tensor, or a callable
        that takes no arguments and returns the actual value to use. The
        exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
        1e-7.
      amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm from
        the paper "On the Convergence of Adam and beyond". Defaults to `False`.
      {{base_optimizer_keyword_args}}
    Reference:
      - [Loshchilov et al., 2019](https://arxiv.org/abs/1711.05101)
      - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980) for `adam`
      - [Reddi et al., 2018](
          https://openreview.net/pdf?id=ryQu7f-RZ) for `amsgrad`.
    Notes:
    The default value of 1e-7 for epsilon might not be a good default in
    general. For example, when training an Inception network on ImageNet a
    current good choice is 1.0 or 0.1. Note that since Adam uses the
    formulation just before Section 2.1 of the Kingma and Ba paper rather than
    the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
    hat" in the paper.
    The sparse implementation of this algorithm (used when the gradient is an
    IndexedSlices object, typically because of `tf.gather` or an embedding
    lookup in the forward pass) does apply momentum to variable slices even if
    they were not used in the forward pass (meaning they have a gradient equal
    to zero). Momentum decay (beta1) is also applied to the entire momentum
    accumulator. This means that the sparse behavior is equivalent to the dense
    behavior (in contrast to some momentum implementations which ignore momentum
    unless a variable slice was actually used).
    """

    def __init__(
        self,
        learning_rate=0.001,
        weight_decay=0.004,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="AdamW",
        **kwargs
    ):
        super().__init__(
            name=name,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            jit_compile=jit_compile,
            **kwargs
        )
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad

        if self.weight_decay is None:
            raise ValueError(
                "Missing value of `weight_decay` which is required and"
                " must be a float value."
            )

    def build(self, var_list):
        """Initialize optimizer variables.
        AdamW optimizer has 3 types of variables: momentums, velocities and
        velocity_hat (only set when amsgrad is applied),
        Args:
          var_list: list of model variables to build AdamW variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="m"
                )
            )
            self._velocities.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="v"
                )
            )
        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(
                    self.add_variable_from_reference(
                        model_variable=var, variable_name="vhat"
                    )
                )

    def _use_weight_decay(self, variable):
        exclude_from_weight_decay = getattr(
            self, "_exclude_from_weight_decay", []
        )
        exclude_from_weight_decay_names = getattr(
            self, "_exclude_from_weight_decay_names", []
        )
        if variable in exclude_from_weight_decay:
            return False
        for name in exclude_from_weight_decay_names:
            if re.search(name, variable.name) is not None:
                return False
        return True

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        beta_1_power = None
        beta_2_power = None
        lr = tf.cast(self.learning_rate, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        beta_1_power = tf.pow(tf.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = tf.pow(tf.cast(self.beta_2, variable.dtype), local_step)

        var_key = self._var_key(variable)
        m = self._momentums[self._index_dict[var_key]]
        v = self._velocities[self._index_dict[var_key]]

        alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        # Apply step weight decay
        if self._use_weight_decay(variable):
            wd = tf.cast(self.weight_decay, variable.dtype)
            variable.assign_sub(variable * wd * lr)

        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            m.assign_add(-m * (1 - self.beta_1))
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1 - self.beta_1), gradient.indices
                )
            )
            v.assign_add(-v * (1 - self.beta_2))
            v.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - self.beta_2),
                    gradient.indices,
                )
            )
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))
        else:
            # Dense gradients.
            m.assign_add((gradient - m) * (1 - self.beta_1))
            v.assign_add((tf.square(gradient) - v) * (1 - self.beta_2))
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "weight_decay": self.weight_decay,
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
            }
        )
        return config

    def exclude_from_weight_decay(self, var_list=None, var_names=None):
        """Exclude variables from weight decays.
        This method must be called before the optimizer's `build` method is
        called. You can set specific variables to exclude out, or set a list of
        strings as the anchor words, if any of which appear in a variable's
        name, then the variable is excluded.
        Args:
            var_list: A list of `tf.Variable`s to exclude from weight decay.
            var_names: A list of strings. If any string in `var_names` appear
                in the model variable's name, then this model variable is
                excluded from weight decay. For example, `var_names=['bias']`
                excludes all bias variables from weight decay.
        """
        if hasattr(self, "_built") and self._built:
            raise ValueError(
                "`exclude_from_weight_decay()` can only be configued before "
                "the optimizer is built."
            )

        self._exclude_from_weight_decay = var_list or []
        self._exclude_from_weight_decay_names = var_names or []



def convert_to_accumulate_gradient_optimizer(orig_optimizer, update_params_frequency, accumulate_sum_or_mean=False):
    if update_params_frequency < 1:
        raise ValueError('update_params_frequency must be >= 1')
    orig_get_gradients = orig_optimizer.get_gradients
    orig_get_updates = orig_optimizer.get_updates
    accumulated_iterations = K.variable(0, dtype='int64', name='accumulated_iterations')
    orig_optimizer.accumulated_iterations = accumulated_iterations

    def updated_get_gradients(self, loss, params):
        return self.accumulate_gradient_accumulators

    def updated_get_updates(self, loss, params):
        self.accumulate_gradient_accumulators = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        updates_accumulated_iterations = K.update_add(accumulated_iterations, 1)
        new_grads = orig_get_gradients(loss, params)
        if not accumulate_sum_or_mean:
            new_grads = [g / K.cast(update_params_frequency, K.dtype(g)) for g in new_grads]
        self.updated_grads = [K.update_add(p, g) for p, g in zip(self.accumulate_gradient_accumulators, new_grads)]
        def update_function():
            with tf.control_dependencies(orig_get_updates(loss, params)):
                reset_grads = [K.update(p, K.zeros(K.int_shape(p), dtype=K.dtype(p))) for p in self.accumulate_gradient_accumulators]
            return tf.group(*(reset_grads + [updates_accumulated_iterations]))
        def just_store_function():
            return tf.group(*[updates_accumulated_iterations])
        
        update_switch = K.equal((updates_accumulated_iterations) % update_params_frequency, 0)
        
        with tf.control_dependencies(self.updated_grads):
            self.updates = [K.switch(update_switch, update_function, just_store_function)]
            return self.updates

    orig_optimizer.get_gradients = updated_get_gradients.__get__(orig_optimizer, type(orig_optimizer))
    orig_optimizer.get_updates = updated_get_updates.__get__(orig_optimizer, type(orig_optimizer))


# Update layers into Keras custom objects
for _name, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isclass(obj) and obj.__module__ == __name__:
        get_custom_objects().update({_name: obj})
