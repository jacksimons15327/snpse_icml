import jax.numpy as jnp
import jax.random as jr
from sbibm.metrics import c2st
import torch
import jax
import matplotlib.pyplot as plt
import time
import os
import random
import numpy as np

from config import get_default_configs
from run import run



import pandas as pd
import itertools

dataset_list = [
    "gaussian_linear_uniform",
    "slcp",
    # "lotka_volterra",
    # "sir",
    "two_moons",
    "gaussian_mixture",
    "gaussian_linear",
    "bernoulli_glm"
]
sbibm_obs_number_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
simulation_budget_list = [1000, 10000, 100000]
batch_size_list = [None]
lr_list = [1e-4]
num_rounds_list = [1,10]
sde_name_list = ["vesde", "vpsde"]
epsilon_list = [1e-3]
max_iters_list = [3000]

list_of_lists = [
    dataset_list,
    sbibm_obs_number_list,
    simulation_budget_list,
    batch_size_list,
    lr_list,
    num_rounds_list,
    sde_name_list,
    epsilon_list,
    max_iters_list,
]
col_names = [
    "dataset", #0
    "sbibm_obs_number", #1
    "simulation_budget", #2
    "batch_size", #3
    "lr", #4
    "num_rounds", #5
    "sde_name", #6
    "epsilon", #7
    "max_iters", #8
]

list_permutations = list(itertools.product(*list_of_lists))
parameter_df = pd.DataFrame(list_permutations)
parameter_df.columns = col_names
parameter_df_full = parameter_df

for jobid in range(68,len(parameter_df_full)):
    jobid = int(jobid)
    key = jr.PRNGKey(jobid)

    parameters = parameter_df_full.iloc[jobid]
    dataset = parameters[0]
    sbibm_obs_number = parameters[1]
    simulation_budget = parameters[2]
    batch_size = parameters[3]
    lr = parameters[4]
    num_rounds = parameters[5]
    sde_name = parameters[6]
    epsilon = parameters[7]
    max_iters = parameters[8]



    config = get_default_configs(dataset = dataset, num_rounds=num_rounds, obs_number=sbibm_obs_number) #0,1,5
    config.algorithm.compute_c2st_intermediate_rounds = False
    config.algorithm.simulation_budget = int(simulation_budget) #2
    config.optim.max_patience = 1000 
    config.optim.lr = lr #4
    config.sde.name = sde_name #5
    config.sampling.epsilon = epsilon #6
    config.optim.max_iters = int(max_iters) #7

    if simulation_budget == 1000:
        t_sample_size = int(10)
        batch_size = int(64)
    elif simulation_budget == 10_000:
        t_sample_size = int(2)
        batch_size = int(512)
    else:
        t_sample_size = int(1)
        batch_size = int(1024)
    config.score_network.t_sample_size = t_sample_size
    config.optim.batch_size = batch_size #3
    config.optim.max_patience = 1000

    config.sampling.n_samples_to_est_boundary = int(1e5)
    config.score_network.use_energy = True


    st = time.time()
    cnf, approx_posterior_samples, c2sts, theta, x, props = run(config, key)
    end = time.time()
    print(f"The run took {end-st} seconds")


    from os.path import exists
    import pandas as pd

    path_for_save = "outputs/"
    if not exists(path_for_save):
        os.system(f"mkdir {path_for_save[:-1]}")

    out_dict = {
            "jobid": jobid,
            "dataset": config.algorithm.dataset, #0
            "sbibm_obs_number": config.algorithm.sbibm_obs_number, #1 
            "simulation_budget": config.algorithm.simulation_budget, #2
            "batch_size": config.optim.batch_size, #3
            "lr": config.optim.lr, #4
            "num_rounds": config.algorithm.num_rounds,#5
            "name": str(config.sde.name),#6
            "epsilon": config.sampling.epsilon,#7
            "max_iters": config.optim.max_iters,#8
            "time_elapsed": end-st,
            "c2st_out": c2sts[-1], 
        }
    out_df = pd.DataFrame(out_dict)
    if not os.path.exists(path_for_save + 'dfs/'):
        os.system(f'mkdir {path_for_save}dfs')
    out_df.to_pickle(path_for_save + "dfs/" + str(jobid) + ".pkl")

    if not exists(path_for_save + "samps/"):
        os.system(f"mkdir {path_for_save}samps")
    np.save(path_for_save + "samps/" + str(jobid) + ".npy", approx_posterior_samples)

    if not exists(path_for_save + "props/"):
        os.system(f"mkdir {path_for_save}props")
    np.save(path_for_save + "props/" + str(jobid) + ".npy", props)

