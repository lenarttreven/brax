import jax
import jax.random as jr
from jax import jacfwd

from brax import envs

env_name = 'halfcheetah'
backend = 'generalized'  # @param ['generalized', 'positional', 'spring']

env = envs.get_environment(env_name=env_name,
                           backend=backend)

state = jax.jit(env.reset)(rng=jr.PRNGKey(seed=0))

u = jr.uniform(key=jr.PRNGKey(seed=0), shape=(env.action_size,))


def step(state, u):
    return env.step(state, u)

out = jacfwd(step, argnums=0)(state, u)

