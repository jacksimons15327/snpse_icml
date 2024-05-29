import sbibm
import jax.numpy as jnp
import torch
import jax
import jax.random as jr

def get_sim_and_prior_from_sbibm(task):
    def prior(num_samples):
        return jnp.array(task.get_prior()(num_samples))

    def simulator(theta):
        theta_numpy = torch.as_tensor(jax.device_get(theta).copy())
        return jnp.array(task.get_simulator()(theta_numpy))
    return simulator, prior

def get_sbcc(cnf, parameter_ds, simulator, config, key):
    n_theta_for_sbcc = 300
    n_samps_per_theta = 300

    th_star = jr.choice(key, parameter_ds, shape=(n_theta_for_sbcc,), replace=False)
    x_star = simulator(th_star)
    props = []
    for ii in range(n_theta_for_sbcc):
        key, lstar_key, sampling_key, l_key = jr.split(key,4)

        l_star = cnf.batch_unn_logp_fn(th_star[ii][:,None].repeat(2,1).T, x_star[ii], lstar_key)[0]

        posterior_x_star_samps = cnf.batch_sample_fn(n_samps_per_theta, x_star[ii], sampling_key)
        l = cnf.batch_unn_logp_fn(posterior_x_star_samps, x_star[ii], l_key)
        props.append((l > l_star).mean().item())
    props.sort()
    return props

def get_sigma_limits(dataset):
    if dataset == "two_moons":
        sigma_max = 3.0
        sigma_min = 0.01
    elif dataset == "gaussian_linear_uniform":
        sigma_max = 5.0
        sigma_min = 0.01
    elif dataset == "gaussian_mixture":
        sigma_max = 10.0
        sigma_min = 0.01
    elif dataset == "slcp":
        sigma_max = 8.0
        sigma_min = 0.05
    elif dataset == "lotka_volterra":
        sigma_max = 12.0
        sigma_min = 0.05
    elif dataset == "sir":
        sigma_max = 2.5
        sigma_min = 0.01
    elif dataset == "gaussian_linear":
        sigma_max = 7
        sigma_min = 0.01
    elif dataset == "bernoulli_glm":
        sigma_max = 10
        sigma_min = 0.01
    else:
        raise NotImplementedError(
            f"sigma_min and sigma_max unknown for task {dataset}."
        )
    return sigma_min, sigma_max