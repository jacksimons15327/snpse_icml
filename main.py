import jax.random as jr
import jax.numpy as jnp
from sbibm.metrics import c2st
import torch
import jax
import matplotlib.pyplot as plt
import time

from config import get_default_configs
from run import run

key = jr.PRNGKey(1)

config = get_default_configs(dataset = "two_moons", simulation_budget = 5000, num_rounds=2, obs_number=1)
config.score_network.use_energy = True

config.optim.max_patience = 500
config.optim.max_iters = 3000
config.optim.lr = 1e-4
config.optim.batch_size = 128
config.sampling.epsilon = 1e-3
config.sampling.n_samples_to_est_boundary = int(1e5)
config.score_network.t_sample_size = 10

st = time.time()
cnf, approx_posterior_samples, c2sts, theta, x, sbcc_props = run(config, key)
print(f"The run took {time.time()-st} seconds")

# # approx_posterior_samples_torch = torch.as_tensor(jax.device_get(approx_posterior_samples).copy(), dtype=torch.float32)
# # c2st_out = c2st(approx_posterior_samples_torch, true_posterior_samples)
# # print(f"The C2ST value is {c2st_out}")

# lin = jnp.linspace(0,1,len(sbcc_props))
# plt.plot(lin, lin)
# plt.plot(lin, sbcc_props)
# plt.show()

# sim_per_round = int(config.algorithm.simulation_budget / config.algorithm.num_rounds)
# for ii in range(config.algorithm.num_rounds):
#     plt.scatter(theta[(ii*sim_per_round):((ii+1)*sim_per_round),0], theta[(ii*sim_per_round):((ii+1)*sim_per_round),1], label=f"round {ii}")
# plt.scatter(approx_posterior_samples[:,0], approx_posterior_samples[:,1], label="est posterior")
# plt.scatter(config.algorithm.posterior_samples_torch[:,0], config.algorithm.posterior_samples_torch[:,1], label="true post")
# plt.legend()
# plt.show()


# true_posterior_samples = config.algorithm.posterior_samples_torch
# key, subkey_logp = jr.split(key)
# true_post_jnp = jnp.array(true_posterior_samples)
# n=100
# upper = 1.5
# lower = -1.5
# grid_points = jnp.linspace(lower,upper,n)
# mesh_grid_points = jnp.meshgrid(grid_points, grid_points)
# theta_array = jnp.stack([mesh_grid_points[0].flatten(), mesh_grid_points[1].flatten()], axis=1)
# log_probs = cnf.batch_logp_fn(theta_array, config.algorithm.x_obs_jnp, key=subkey_logp)
# log_probs = jnp.where(log_probs < -10, -10, log_probs)
# plt.figure(figsize=(16, 12))
# plt.imshow(log_probs.reshape(n,n), cmap='viridis', extent=[lower, upper, lower, upper], origin='lower')
# plt.scatter(approx_posterior_samples[:,0], approx_posterior_samples[:,1], label='Est Samples')
# plt.scatter(true_post_jnp[:,0], true_post_jnp[:,1], label='True Samples')
# plt.legend()
# plt.show()






# # # Plot the pairplots in the subplots
# # import matplotlib.pyplot as plt
# # from sbi.analysis import pairplot

# # # Create a figure with 2 subplots
# # fig, axs = plt.subplots(2, 1, figsize=(4, 8))
# # _ = pairplot(approx_posterior_samples_torch, limits=[[-3, 3], [-3, 3], [-3, 3], [-3, 3], [-3, 3]])
# # _ = pairplot(true_posterior_samples, limits=[[-3, 3], [-3, 3], [-3, 3], [-3, 3], [-3, 3]])

# # plt.tight_layout()
# # plt.show()