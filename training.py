import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import functools as ft
import optax
import math
from copy import deepcopy

def train_score_network(config, model, sde, parameter_ds, data_ds, key):
    split_indices = [parameter_ds.shape[1], parameter_ds.shape[1] + data_ds.shape[1]]
    train_dataloader_key, eval_dataloader_key, train_loss_key, eval_loss_key, data_split_key = jr.split(key, 5)

    weight = get_weight(config, sde)
    
    ds = jnp.concatenate([parameter_ds, data_ds], axis=1)
    ds_mean = ds.mean(axis=0)
    ds_std = ds.std(axis=0)
    ds = (ds - ds_mean) / ds_std

    indices = jr.permutation(data_split_key, jnp.arange(ds.shape[0]))

    split_index = int(ds.shape[0] * config.optim.eval_prop)
    train_ds = ds[indices[split_index:]]
    eval_ds = ds[indices[:split_index]]

    opt = optax.adam(config.optim.lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    total_train_loss = 0; total_eval_loss = 0
    total_train_size = 0; total_eval_size = 0
    best_loss = float("inf")
    patience = 0
    for step in range(config.optim.max_iters):  

        # training
        for _ in range(config.score_network.t_sample_size):
            for train_ds_batch in dataloader(train_ds, config.optim.batch_size, key=train_dataloader_key):
                train_batch_theta, train_batch_x, _ = jnp.split(train_ds_batch, indices_or_sections=split_indices, axis=1)
                # train_batch_theta, train_batch_x, _ = jnp.split(train_ds, indices_or_sections=split_indices, axis=1)
                train_loss, model, train_loss_key, opt_state = make_step(
                    model, sde, weight, train_batch_theta, train_batch_x, train_loss_key, opt_state, opt.update
                )
                total_train_loss += train_loss.item()
                total_train_size += 1
    
        # evaluation
        for _ in range(config.score_network.t_sample_size):
            for eval_ds_batch in dataloader(eval_ds, config.optim.batch_size, key=eval_dataloader_key):
                eval_batch_theta, eval_batch_x, _ = jnp.split(eval_ds_batch, indices_or_sections=split_indices, axis=1)
                # eval_batch_theta, eval_batch_x, _ = jnp.split(eval_ds, indices_or_sections=split_indices, axis=1)
                eval_loss, eval_loss_key = do_eval(
                    model, sde, weight, eval_batch_theta, eval_batch_x, eval_loss_key
                )
                total_eval_loss += eval_loss.item()
                total_eval_size += 1

        if math.isnan(eval_loss.item()) or math.isnan(train_loss.item()):
            print(eval_loss, train_loss, eval_batch_theta, train_batch_theta)
            print(f"Step={step}: training loss={round(total_train_loss / total_train_size,3)}, evaluation loss={round(total_eval_loss / total_eval_size,3)}, patience = {patience}")
            print("NaN loss encountered. Aborting training.")
            break

        train_loss_avg = total_train_loss / total_train_size
        eval_loss_avg = total_eval_loss / total_eval_size

        if eval_loss_avg < best_loss:
            best_model = deepcopy(model)
            best_loss = eval_loss_avg
            patience = 0
        else:
            patience += 1
            if patience > config.optim.max_patience:
                print(f"Early stopping at step {step} with evaluation loss {round(best_loss,3)}")
                break
            
        # print
        if (step % config.optim.print_every) == 0 or (step == config.optim.max_iters - 1):
            print(f"Step={step}: training loss={round(train_loss_avg,3)}, evaluation loss={round(eval_loss_avg,3)}, patience = {patience}, best_loss={round(best_loss,3)}")
        total_train_loss = 0
        total_train_size = 0
        total_eval_loss = 0
        total_eval_size = 0


    return best_model, ds_mean, ds_std

def single_loss_fn(model, sde, weight, theta, x, t, key):
    mean, std = sde.marginal_prob(theta, t)
    noise = jr.normal(key, theta.shape)
    pert_theta = mean + std * noise
    pred = model(pert_theta, x, std)
    return weight(t) * jnp.mean((pred + noise / std) ** 2)

def batch_loss_fn(model, sde, weight, theta, x, key):
    batch_size = theta.shape[0]
    tkey, losskey = jr.split(key)
    losskey = jr.split(losskey, batch_size)
    
    # Low-discrepancy sampling over t to reduce variance
    # t = jr.uniform(tkey, (batch_size,), minval=0.0001, maxval=sde.T / batch_size)
    # t = t + (sde.T / batch_size) * jnp.arange(batch_size)

    t = jr.uniform(tkey, (batch_size,), minval=0.0001, maxval=sde.T)

    loss_fn = ft.partial(single_loss_fn, model, sde, weight)
    loss_fn = jax.vmap(loss_fn)
    return jnp.mean(loss_fn(theta, x, t, losskey))

@eqx.filter_jit
def make_step(model, sde, weight, theta, x, key, opt_state, opt_update):
    loss_fn = eqx.filter_value_and_grad(batch_loss_fn)
    loss, grads = loss_fn(model, sde, weight, theta, x, key)
    updates, opt_state = opt_update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    key = jr.split(key, 1)[0]
    return loss, model, key, opt_state

@eqx.filter_jit
def do_eval(model, sde, weight, theta, x, key):
    loss_fn = eqx.filter_value_and_grad(batch_loss_fn)
    loss, _ = loss_fn(model, sde, weight, theta, x, key)
    key = jr.split(key, 1)[0]
    return loss, key

def get_weight(config, sde):
    if config.score_network.use_weighted_loss:
        def weight(t):
            _, std = sde.marginal_prob(jnp.zeros(config.algorithm.task.dim_parameters), t)
            return std**2
    else:
        def weight(t):
            return jnp.ones_like(t)
    return weight


def dataloader(data, batch_size, *, key):
    dataset_size = data.shape[0]
    indices = jnp.arange(dataset_size)
    key, subkey = jr.split(key)
    perm = jr.permutation(subkey, indices)
    start = 0
    while start < dataset_size:
        end = min(start + batch_size, dataset_size)
        batch_perm = perm[start:end]
        yield data[batch_perm]
        start = end