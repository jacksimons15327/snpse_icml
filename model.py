import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax  # https://github.com/deepmind/optax

class NCMLP(eqx.Module):
    mlp_theta: eqx.nn.MLP
    mlp_x: eqx.nn.MLP
    mlp_main: eqx.nn.MLP
    t_embed_dim: int

    def __init__(self, key, config):
        key1, key2, key3 = jr.split(key, 3)
        self.mlp_theta = eqx.nn.MLP(
            in_size = config.algorithm.task.dim_parameters,
            out_size = config.score_network.theta_embed_dim, 
            depth = config.score_network.depth,
            width_size = config.score_network.width,
            activation = config.score_network.activation,
            key = key1)
        self.mlp_x = eqx.nn.MLP(
            in_size = config.algorithm.task.dim_data,
            out_size = config.score_network.x_embed_dim, 
            depth = config.score_network.depth,
            width_size = config.score_network.width,
            activation = config.score_network.activation,
            key = key2)
        self.mlp_main = eqx.nn.MLP(
            in_size = config.score_network.theta_embed_dim + config.score_network.x_embed_dim + config.score_network.t_embed_dim,
            out_size = config.algorithm.task.dim_parameters, 
            depth = config.score_network.depth,
            width_size = config.score_network.width,
            activation = config.score_network.activation,
            key = key3)
        self.t_embed_dim = config.score_network.t_embed_dim

        # if config.score_network.use_layer_norm:
        #     self.norm_x = eqx.nn.LayerNorm(config.score_network.x_embed_dim)
        #     self.norm_theta = eqx.nn.LayerNorm(config.score_network.theta_embed_dim)
        #     self.norm_main = eqx.nn.LayerNorm(config.algorithm.task.dim_data)

    def __call__(self, theta, x, sigma):
        # T EMB
        # t_emb = jnp.tile(sigma, (self.t_embed_dim,))
        t_emb = get_timestep_embedding(sigma, self.t_embed_dim).squeeze()

        # THETA EMB
        theta_emb = self.mlp_theta(theta)

        # X EMB
        x_emb = self.mlp_x(x)

        # MAIN
        out = jnp.concatenate([theta_emb, x_emb, t_emb])
        out = self.mlp_main(out)

        return out

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    half_dim = embedding_dim // 2
    emb = jnp.log(max_positions) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
    emb = timesteps.ravel().astype(jnp.float32)[:,None] * emb
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    return emb