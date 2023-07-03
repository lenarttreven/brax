import functools
from datetime import datetime

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from flax import struct

from brax import envs
from brax.training.agents.sac import train as sac

env_name = 'pendulum'
backend = 'spring'  # @param ['generalized', 'positional', 'spring']

env = envs.get_environment(env_name=env_name,
                           backend=backend)
state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))

train_fn = {
    'pendulum': functools.partial(sac.train, num_timesteps=30_000, num_evals=20, reward_scaling=1,
                                  episode_length=100, normalize_observations=True, action_repeat=1,
                                  discounting=0.999, learning_rate=3e-4, num_envs=16, batch_size=64,
                                  grad_updates_per_step=32, max_replay_size=10 ** 5,
                                  min_replay_size=2 ** 8, seed=1, num_eval_envs=1, deterministic_eval=True,
                                  tau=0.005)
}[env_name]

max_y = 0
min_y = -100

xdata, ydata = [], []
times = [datetime.now()]


def progress(num_steps, metrics):
    times.append(datetime.now())
    xdata.append(num_steps)
    ydata.append(metrics['eval/episode_reward'])
    plt.xlim([0, train_fn.keywords['num_timesteps']])
    plt.ylim([min_y, max_y])
    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.plot(xdata, ydata)
    plt.show()


make_inference_fn, params, metrics = train_fn(environment=env, progress_fn=progress)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')


def policy(x, params, key):
    return make_inference_fn(params, deterministic=True)(x, key)[0]


init_state = jnp.array([jnp.pi, 0.0])
dt = 0.1
T = 100


def _ode(x, u):
    assert x.shape == (2,) and u.shape == (1,)
    system_params = jnp.array([5.0, 9.81])
    u = 4 * u
    return jnp.array([x[1], system_params[1] / system_params[0] * jnp.sin(x[0]) + u.reshape()])


def convert_angle(x):
    return jnp.arctan2(jnp.sin(x), jnp.cos(x))


def next_step(x: chex.Array, params):
    u = policy(x, params, jr.PRNGKey(0))
    x_next = x + dt * _ode(x, u)
    x_next = x_next.at[0].set(convert_angle(x_next[0]))
    return x_next, 4 * u


@struct.dataclass
class StateCarry:
    x: chex.Array
    params: chex.Array


from jax.lax import scan


def f(carry: StateCarry, _):
    x_next, u = next_step(carry.x, carry.params)
    carry = carry.replace(x=x_next)
    return carry, (carry.x, u)


new_carry, (xs, us) = scan(f, StateCarry(x=init_state, params=params), xs=None, length=T)
ts = jnp.linspace(0, T, 100)
plt.plot(ts, xs, label='xs')
plt.plot(ts, us, label='us')
plt.legend()
plt.show()
