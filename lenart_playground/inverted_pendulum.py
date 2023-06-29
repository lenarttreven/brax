import functools
from datetime import datetime

import jax
import matplotlib.pyplot as plt

from brax import envs
from brax.training.agents.sac import train as sac

env_name = 'inverted_pendulum'
backend = 'positional'  # @param ['generalized', 'positional', 'spring']

env = envs.get_environment(env_name=env_name,
                           backend=backend)
state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))

train_fn = {
    'inverted_pendulum': functools.partial(sac.train, num_timesteps=300_000, num_evals=20, reward_scaling=10,
                                           episode_length=1000, normalize_observations=True, action_repeat=1,
                                           discounting=0.97, learning_rate=3e-4, num_envs=64, batch_size=64,
                                           grad_updates_per_step=32, max_replay_size=2 ** 14,
                                           min_replay_size=2 ** 9, seed=1, num_eval_envs=32)
}[env_name]

max_y = 2000
min_y = 0

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


make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')
