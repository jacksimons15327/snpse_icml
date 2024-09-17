import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import torch

from task_utils import get_sim_and_prior_from_sbibm
from model import NCMLP
from model_energy import NCMLP_ENERGY
from model_energy_resnet import NCResnet
from sde import get_sde
from training import train_score_network
from cnf import CNF
from task_utils import get_sbcc
from sampling import get_c2st, get_truncated_prior, get_truncated_prior_energy

def run(config, key):
    c2sts = []
    sde = get_sde(config)
    simulator, prior = get_sim_and_prior_from_sbibm(config.algorithm.task)
    sims_per_round = int(config.algorithm.simulation_budget / config.algorithm.num_rounds)

    rr = 0
    for rr in range(config.algorithm.num_rounds):
        if rr == 0:
            parameter_ds = prior(sims_per_round)
            data_ds = simulator(parameter_ds)
        else:
            key, subkey_sample = jr.split(key)
            if config.score_network.use_energy:
                truncated_prior = get_truncated_prior_energy(cnf, config, prior, key)
            else:
                truncated_prior = get_truncated_prior(cnf, config, prior, key)
            trunc_prior_samps = truncated_prior(sims_per_round, subkey_sample)
            parameter_ds = jnp.concatenate([parameter_ds, trunc_prior_samps], axis=0)
            data_ds = jnp.concatenate([data_ds, simulator(trunc_prior_samps)], axis=0)

        key, subkey_model, subkey_training = jr.split(key, 3)
        
        if config.score_network.use_energy:
            if config.resnet.use:
                model = NCResnet(subkey_model, config)
            else:
                model = NCMLP_ENERGY(subkey_model, config)
        else:
            model = NCMLP(subkey_model, config)
        model, ds_means, ds_stds = train_score_network(config, model, sde, parameter_ds, data_ds, subkey_training)

        cnf = CNF(
            score_network=model,
            sde=sde, 
            ds_means=ds_means, 
            ds_stds=ds_stds, 
            )

        if (rr==(config.algorithm.num_rounds-1)) or config.algorithm.compute_c2st_intermediate_rounds:
            key, subkey_c2st_sample = jr.split(key)
            c2st_out, post_samps = get_c2st(cnf, config, subkey_c2st_sample)
            c2sts.append(c2st_out)
            print(f"The C2ST value in round {rr+1} is {c2st_out}")

    sbcc_props = get_sbcc(cnf, parameter_ds, simulator, config, key)

    return cnf, post_samps, c2sts, parameter_ds, data_ds, sbcc_props
        


