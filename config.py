import ml_collections
import sbibm
import equinox as eqx
import jax
import jax.numpy as jnp
from task_utils import get_sigma_limits

def get_default_configs(
    simulation_budget, num_rounds, dataset, obs_number
):
    config = ml_collections.ConfigDict()

    # algorithm
    config.algorithm = algorithm = ml_collections.ConfigDict()
    algorithm.dataset = dataset
    algorithm.sbibm_obs_number = obs_number
    algorithm.simulation_budget = simulation_budget
    algorithm.task = sbibm.get_task(dataset)
    algorithm.num_rounds = int(num_rounds)
    algorithm.compute_c2st_intermediate_rounds = False
    algorithm.x_obs = algorithm.task.get_observation(obs_number)
    algorithm.x_obs_jnp = jnp.array(algorithm.task.get_observation(obs_number)).reshape(-1,)
    algorithm.posterior_samples_torch = algorithm.task.get_reference_posterior_samples(obs_number)

    # score_network
    config.score_network = score_network = ml_collections.ConfigDict()
    score_network.use_energy = True
    score_network.width = 256
    score_network.depth = 2
    score_network.activation = jax.nn.silu
    score_network.t_embed_dim = 64
    score_network.theta_embed_dim = max(30, 4*algorithm.task.dim_parameters)
    score_network.x_embed_dim = max(30, 4*algorithm.task.dim_data)
    score_network.use_weighted_loss = True
    score_network.t_sample_size = 10
    # score_network.use_layer_norm = False

    # sde
    config.sde = sde = ml_collections.ConfigDict()
    sde.name = "vpsde"
    sde.T = 1.0
    sde.beta_min = 0.1
    sde.beta_max = 10.0
    sigma_min, sigma_max = get_sigma_limits(dataset)
    sde.sigma_min = sigma_min
    sde.sigma_max = sigma_max

    # optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.max_iters = 1000
    optim.batch_size = 64 if simulation_budget <= 1000 else 512
    optim.eval_prop = 0.10
    optim.lr = 1e-4
    optim.print_every = 200
    optim.max_patience = 250

    #sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.epsilon = 5e-4
    sampling.n_samples_to_est_boundary = int(1e5)



    return config
