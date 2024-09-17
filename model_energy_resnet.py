import equinox as eqx
import jax
import jax.numpy as jnp
import functools as ft
import jax.random as jr
import optax  # https://github.com/deepmind/optax

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

class Resnet(eqx.Module):
    layers: list
    embed_dim: int
    layer_norm: eqx.nn.LayerNorm

    def __init__(self, key, config):
        num_blocks = config.resnet.num_blocks
        keys = jr.split(key, config.resnet.num_blocks)
        embed_dim = config.resnet.theta_embed_dim
        self.layers = [ResnetBlock(key, config) for key in keys]
        self.embed_dim = embed_dim
        self.layer_norm = eqx.nn.LayerNorm(embed_dim)


    def __call__(self, theta, context_x, context_t):
        for layer in self.layers:
            theta = layer(theta, context_x, context_t)

        theta = self.layer_norm(theta)
        return theta

class ResnetBlock(eqx.Module):
    layer_norm1: eqx.nn.LayerNorm
    layer_norm2: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP
    act: jax.nn.silu
    layer_context_t: eqx.nn.Linear
    layer_context_x: eqx.nn.Linear

    def __init__(self, key, config):
        key_mlp, key_emb_x, key_emb_t = jr.split(key, 3)
        embed_dim = config.resnet.theta_embed_dim
        widening_factor = config.resnet.widening_factor

        self.layer_norm1 = eqx.nn.LayerNorm(embed_dim)
        self.layer_norm2 = eqx.nn.LayerNorm(embed_dim)

        self.mlp = eqx.nn.MLP(
            in_size = embed_dim, 
            out_size = embed_dim,
            width_size = embed_dim * widening_factor,
            depth = 1,
            activation = config.resnet.activation,
            key = key_mlp
            )
        self.layer_context_x = eqx.nn.Linear(
            in_features = config.resnet.x_embed_dim, 
            out_features = embed_dim,
            key = key_emb_x)
        self.layer_context_t = eqx.nn.Linear(
            in_features = config.resnet.t_embed_dim, 
            out_features = embed_dim,
            key = key_emb_t)
        self.act = config.resnet.activation

    def __call__(self, theta, context_x, context_t):
        h = self.layer_norm1(theta)
        
        #get embedded x
        context_x_emb = self.layer_context_x(context_x)
        context_x_emb = self.act(context_x_emb)

        #condition on x
        h += context_x_emb
        # h = self.layer_norm2(h) 

        # mlp
        h = self.mlp(h)
        
        #get embedded t
        context_t_emb = self.layer_context_t(context_t)
        context_t_emb = self.act(context_t_emb)
        
        #condition on t
        h += context_t_emb

        return theta + h

class NCResnet(eqx.Module):
    linear_th: eqx.nn.Linear
    linear_x: eqx.nn.Linear
    t_embed_dim: int
    resnet: Resnet
    final_layer: eqx.nn.Linear
    embed_1: eqx.nn.Linear

    def __init__(self, key, config):
        key_x, key_theta, key_main, key_other, key_final = jr.split(key, 5)

        self.linear_x = eqx.nn.Linear(
            config.algorithm.task.dim_data,
            config.resnet.x_embed_dim, 
            key = key_x)
        self.linear_th = eqx.nn.Linear(
            config.algorithm.task.dim_parameters,
            config.resnet.theta_embed_dim, 
            key = key_theta)
        self.t_embed_dim = config.resnet.t_embed_dim
        self.resnet = Resnet(
            key = key_main,
            config = config,
        )
        self.final_layer = eqx.nn.Linear(
            in_features = config.resnet.theta_embed_dim,
            out_features = config.algorithm.task.dim_parameters,
            key = key_other
        )
        self.embed_1 = eqx.nn.Linear(
            in_features = config.algorithm.task.dim_parameters,
            out_features = 1,
            key = key_final
        )

    def energy(self, theta, x, sigma):
        # THETA EMB
        theta_emb = self.linear_th(theta)

        # X EMB
        x_emb = self.linear_x(x)

        # T EMB
        t_emb = get_timestep_embedding(sigma, self.t_embed_dim).squeeze()

        # MAIN
        out = self.resnet(theta=theta_emb, context_x=x_emb, context_t=t_emb)

        out = self.final_layer(out)

        out += theta

        out_1 = self.embed_1(out)

        return out_1.squeeze()

    def __call__(self, theta, x, sigma):
        energy_fn = ft.partial(self.energy, x=x, sigma=sigma)
        score_fn = jax.grad(energy_fn)
        return score_fn(theta)

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    half_dim = embedding_dim // 2
    emb = jnp.log(max_positions) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
    emb = timesteps.ravel().astype(jnp.float32)[:,None] * emb
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    return emb
