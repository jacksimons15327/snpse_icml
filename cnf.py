from model import NCMLP
from sde import SDE

import math
import functools as ft

import diffrax as dfx
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.numpy as jnp
import jax.random as jr
import optax  # https://github.com/deepmind/optax


class CNF(eqx.Module):
    score_network: NCMLP
    sde: SDE
    parameter_mean: jnp.ndarray
    parameter_std: jnp.ndarray
    data_mean: jnp.ndarray
    data_std: jnp.ndarray
    t1: float
    t0: float
    dt: float


    def __init__(
        self,
        *,
        score_network,
        sde,
        ds_means,
        ds_stds,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.score_network = score_network
        self.sde = sde
        self.t1 = sde.T
        self.t0 = 1e-5
        self.dt = 0.01

        split_indices = [self.sde.dim_parameters, self.sde.dim_parameters + self.sde.dim_data]
        parameter_mean, data_mean, _ = jnp.split(ds_means, indices_or_sections=split_indices)
        parameter_std, data_std, _ = jnp.split(ds_stds, indices_or_sections=split_indices)
        self.parameter_mean = parameter_mean
        self.parameter_std = parameter_std
        self.data_mean = data_mean
        self.data_std = data_std

    @eqx.filter_jit
    def batch_sample_fn(self, sample_size, x, key):
        x = (x - self.data_mean) / self.data_std
        sample_keys = jr.split(key, sample_size)
        sample_fn = ft.partial(self.single_sample_fn, self.score_network, self.sde, x)
        samples = jax.vmap(sample_fn)(sample_keys)
        samples = self.parameter_mean + self.parameter_std * samples
        return samples

    @eqx.filter_jit
    def single_sample_fn(self, score_network, sde, x, key, epsilon=1e-5, dt=0.01):
        key, base_dist_key = jr.split(key)

        drift = ft.partial(sde.drift_ode, score_network, x)
        term = dfx.ODETerm(drift)
        
        init_theta = sde.base_dist(base_dist_key).reshape(-1,)

        # solver = dfx.Dopri5()
        solver = dfx.Tsit5()
        sol = dfx.diffeqsolve(term, solver, sde.T, epsilon, -dt, init_theta)
        return sol.ys[0]

    @eqx.filter_jit
    def batch_logp_fn(self, theta, x, key):
        x = (x - self.data_mean) / self.data_std
        theta = (theta - self.parameter_mean) / self.parameter_std
        logp_keys = jr.split(key, theta.shape[0])
        logp_fn = ft.partial(self.single_logp_fn, self.score_network, self.sde, x)
        logps = jax.vmap(logp_fn)(theta, logp_keys)
        return logps

    @eqx.filter_jit
    def single_logp_fn(self, score_network, sde, x, theta, key):
        term = ft.partial(sde.drift_dlogp_ode, score_network, x)
        term = dfx.ODETerm(term)

        delta_log_likelihood = 0.0
        theta = (theta, delta_log_likelihood)

        # solver = dfx.Dopri5()
        solver = dfx.Tsit5()
        sol = dfx.diffeqsolve(term, solver, self.t0, self.t1, self.dt, theta)
        (y,), (delta_log_likelihood,) = sol.ys
        return delta_log_likelihood + sde.base_dist_logp(y)

    @eqx.filter_jit
    def batch_unn_logp_fn(self, theta, x, key):
        x = (x - self.data_mean) / self.data_std
        theta = (theta - self.parameter_mean) / self.parameter_std
        logp_keys = jr.split(key, theta.shape[0])
        logp_fn = ft.partial(self.single_unn_logp_fn, x)
        logps = jax.vmap(logp_fn)(theta, logp_keys)
        return logps

    @eqx.filter_jit
    def single_unn_logp_fn(self, x, theta, key):
        _, sigma = self.sde.marginal_prob(theta, jnp.array(self.t0))
        energy = self.score_network.energy(theta, x, sigma)
        return energy



