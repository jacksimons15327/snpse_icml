import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import functools as ft
from tqdm import tqdm
from sbibm.metrics import c2st
import torch

def get_truncated_prior(cnf, config, prior, key):
    est_posterior_samples = cnf.batch_sample_fn(int(config.sampling.n_samples_to_est_boundary), config.algorithm.x_obs_jnp, key)
    log_probs = cnf.batch_logp_fn(est_posterior_samples, config.algorithm.x_obs_jnp, key)
    trunc_boundary = jnp.quantile(log_probs, config.sampling.epsilon)
    # trunc_boundary = jnp.quantile(log_probs, 1e-3)
    posterior_uniform_hypercube_min = est_posterior_samples.min(axis=0)
    posterior_uniform_hypercube_max = est_posterior_samples.max(axis=0)
    prior_uniform_hypercube_min = prior(int(1e6)).min(axis=0)
    prior_uniform_hypercube_max = prior(int(1e6)).max(axis=0)
    hypercube_min = jnp.concatenate([posterior_uniform_hypercube_min[None,:], prior_uniform_hypercube_min[None,:]], axis=0).max(axis=0)
    hypercube_max = jnp.concatenate([posterior_uniform_hypercube_max[None,:], prior_uniform_hypercube_max[None,:]], axis=0).min(axis=0)
    def hypercube_uniform_prior(num_samples, key):
        return jax.random.uniform(key, (num_samples, len(prior_uniform_hypercube_min)), minval=hypercube_min, maxval=hypercube_max)

    def truncated_prior(num_samples, key):
        max_iters = 1000
        counter = 0
        n_samples_so_far = 0
        samples_out = []
        while (n_samples_so_far < num_samples) and (counter < max_iters):
            key, subkey_prior, subkey_logp = jr.split(key, 3)
            samples = hypercube_uniform_prior(num_samples, subkey_prior)
            log_probs = cnf.batch_logp_fn(samples, config.algorithm.x_obs_jnp, subkey_logp)
            accepted_samples = samples[log_probs > trunc_boundary]
            samples_out.append(accepted_samples)
            n_samples_so_far += len(accepted_samples)
            counter+=1

        if counter == max_iters:
            assert ValueError("Truncated prior sampling did not converge in the allowed number of iterations - returning error.")
        return jnp.concatenate(samples_out)[0:num_samples]
    
    return truncated_prior

def get_truncated_prior_energy(cnf, config, prior, key):
    est_posterior_samples = cnf.batch_sample_fn(int(config.sampling.n_samples_to_est_boundary), config.algorithm.x_obs_jnp, key)
    log_probs = cnf.batch_unn_logp_fn(est_posterior_samples, config.algorithm.x_obs_jnp, key)
    trunc_boundary = jnp.quantile(log_probs, config.sampling.epsilon)
    
    def truncated_prior(num_samples, key):
        max_iters = 100_000
        counter = 0
        n_samples_so_far = 0
        samples_out = []
        while (n_samples_so_far < num_samples) and (counter < max_iters):
            key, subkey_logp = jr.split(key, 2)
            samples = prior(num_samples)
            log_probs = cnf.batch_unn_logp_fn(samples, config.algorithm.x_obs_jnp, subkey_logp)
            accepted_samples = samples[log_probs > trunc_boundary]
            samples_out.append(accepted_samples)
            n_samples_so_far += len(accepted_samples)
            counter+=1

        if counter == max_iters:
            assert ValueError("Truncated prior sampling did not converge in the allowed number of iterations - returning error.")
        return jnp.concatenate(samples_out)[0:num_samples]
    
    return truncated_prior


def get_c2st(cnf, config, key):
    est_posterior_samples = cnf.batch_sample_fn(10000, config.algorithm.x_obs_jnp, key)
    approx_posterior_samples_torch = torch.as_tensor(jax.device_get(est_posterior_samples).copy(), dtype=torch.float32)
    c2st_out = c2st(approx_posterior_samples_torch, config.algorithm.posterior_samples_torch)

    return c2st_out, est_posterior_samples