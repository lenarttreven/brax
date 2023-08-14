import functools
from datetime import datetime

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from flax import struct

from brax import envs
from brax.training.agents.ppo import train as ppo

env_name = 'pendulum'
backend = 'spring'  # @param ['generalized', 'positional', 'spring']

env = envs.get_environment(env_name=env_name, backend=backend)
state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))

train_fn = {
    'pendulum': functools.partial(ppo.train, num_timesteps=200_000, num_evals=20, episode_length=100, reward_scaling=10,
                                  normalize_observations=True, unroll_length=50, num_minibatches=32,
                                  action_repeat=1, learning_rate=3e-4, num_envs=16, seed=1, num_eval_envs=1,
                                  num_updates_per_batch=4, discounting=0.99, entropy_cost=1e-1,
                                  batch_size=64, deterministic_eval=True)
}[env_name]

max_y = 0
min_y = -100

xdata, ydata = [], []
times = [datetime.now()]


def progress(num_steps, metrics):
    times.append(datetime.now())
    xdata.append(num_steps)
    ydata.append(metrics['eval/episode_reward'])
    # plt.xlim([0, train_fn.keywords['num_timesteps']])
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
    u_enlarged = policy(x, params, jr.PRNGKey(0))
    u, eta = u_enlarged[:1], u_enlarged[1:]
    x_next = x + dt * _ode(x, u)
    x_next = x_next.at[0].set(convert_angle(x_next[0]))
    return x_next, (4 * u, eta)



@struct.dataclass
class StateCarry:
    x: chex.Array
    params: chex.Array


from jax.lax import scan


def f(carry: StateCarry, _):
    x_next, (u, eta) = next_step(carry.x, carry.params)
    carry = carry.replace(x=x_next)
    return carry, (carry.x, u, eta)


# def f_hal(carry: StateCarry, _):
#     x_next, (u, eta) = next_step_halucination(carry.x, carry.params)
#     carry = carry.replace(x=x_next)
#     return carry, (carry.x, u, eta)


# Evaluate on the true system
new_carry, (xs, us, etas) = scan(f, StateCarry(x=init_state, params=params), xs=None, length=T)
ts = jnp.linspace(0, T, 100)
plt.plot(ts, xs, label='xs')
plt.plot(ts, us, label='us')
plt.plot(ts, etas, label='etas')
plt.legend()
plt.title('True system')
plt.show()
#
# # Halucinate
# new_carry, (xs, us, etas) = scan(f_hal, StateCarry(x=init_state, params=params), xs=None, length=T)
# ts = jnp.linspace(0, T, 100)
# plt.plot(ts, xs, label='xs')
# plt.plot(ts, us, label='us')
# plt.plot(ts, etas, label='etas')
# plt.legend()
# plt.title('Halucinated system')
# plt.show()
