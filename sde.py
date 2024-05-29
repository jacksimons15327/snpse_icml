import jax.numpy as jnp
import jax.random as jr
import jax

def get_sde(config):
    if config.sde.name == "vesde":
        return VESDE(config)
    elif config.sde.name == "vpsde":
        return VPSDE(config)
    else:
        raise NotImplementedError(f"SDE {config.sde.name} not implemented.")


class SDE():
    def __init__(self, config):
        self.T = config.sde.T
        self.dim_data = config.algorithm.task.dim_data
        self.dim_parameters = config.algorithm.task.dim_parameters
        self.beta_min = config.sde.beta_min
        self.beta_max = config.sde.beta_max
        self.sigma_min = config.sde.sigma_min
        self.sigma_max = config.sde.sigma_max

    def marginal_prob(self, theta, t):
        raise NotImplementedError

    def drift_ode(self, score_model, x, t, theta, args):
        raise NotImplementedError

    def base_dist(self, key):
        raise NotImplementedError

    def drift_dlogp_ode(self, score_model, x, t, theta, args):
        raise NotImplementedError


class VPSDE(SDE):
    def __init__(self, config):
        super().__init__(config)

    def marginal_prob(self, theta, t):
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        )
        mean = jnp.exp(log_mean_coeff) * theta
        std = jnp.sqrt(1.0 - jnp.exp(2.0 * log_mean_coeff))
        return mean, std

    def drift_ode(self, score_model, x, t, theta, args):
        beta_t = self.beta_min + t*(self.beta_max - self.beta_min)
        _, std = self.marginal_prob(theta, t)
        return -0.5 * beta_t * (theta + score_model(theta, x, std))

    def base_dist(self, key):
        return jr.normal(key, (1, self.dim_parameters))
    
    def base_dist_logp(self, theta):
        return jax.scipy.stats.multivariate_normal.logpdf(theta, mean=jnp.zeros(self.dim_parameters,), cov=jnp.eye(self.dim_parameters,self.dim_parameters))

    def drift_dlogp_ode(self, score_model, x, t, theta, args):
        # *args, eps = args
        theta, _ = theta

        drift_fn = lambda y: self.drift_ode(score_model, x, t, y, args)
        drift, vjp_fn = jax.vjp(drift_fn, theta)

        (size,) = theta.shape 
        eye = jnp.eye(size)
        (dfdtheta,) = jax.vmap(vjp_fn)(eye)
        dlogp = jnp.trace(dfdtheta)

        # drift = -0.5 * beta_t * (x + fn)
        return (drift, dlogp)


class VESDE(SDE):
    def __init__(self, config):
        super().__init__(config)

    def marginal_prob(self, theta, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = theta
        return mean, std

    def base_dist(self, key):
        return jr.normal(key, (1, self.dim_parameters))*self.sigma_max
    
    def base_dist_logp(self, theta):
        return jax.scipy.stats.multivariate_normal.logpdf(theta, mean=jnp.zeros(self.dim_parameters,), cov=jnp.eye(self.dim_parameters,self.dim_parameters)*self.sigma_max**2)

    def drift_ode(self, score_model, x, t, theta, args):
        _, std = self.marginal_prob(theta, t)
        g_sq = (std**2) * (2 * (jnp.log(self.sigma_max) - jnp.log(self.sigma_min)))
        return -0.5 * g_sq * score_model(theta, x, std)

    def drift_dlogp_ode(self, score_model, x, t, theta, args):
        # *args, eps = args
        theta, _ = theta

        drift_fn = lambda y: self.drift_ode(score_model, x, t, y, args)
        drift, vjp_fn = jax.vjp(drift_fn, theta)

        (size,) = theta.shape 
        eye = jnp.eye(size)
        (dfdtheta,) = jax.vmap(vjp_fn)(eye)
        dlogp = jnp.trace(dfdtheta)

        # drift = -0.5 * beta_t * (x + fn)
        return (drift, dlogp)