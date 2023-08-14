import chex
from jax import numpy as jnp

from brax.envs.base import State, Env


class Pendulum(Env):

    def __init__(self,
                 backend: str = 'string',
                 params: chex.Array = jnp.array([0.1, 5.0])):
        self.dt = params[0]
        self.T_horizon = 10
        self._backend = backend
        self.length = params[1]

    def _ode(self, x, u):
        assert x.shape == (2,) and u.shape == (1,)
        g = 9.81
        u = 4 * u
        return jnp.array([x[1], g / self.length * jnp.sin(x[0]) + u.reshape()])

    def convert_angle(self, x):
        return jnp.arctan2(jnp.sin(x), jnp.cos(x))

    def reward(self, x, u):
        assert x.shape == (2,) and u.shape == (1,)
        u = 4 * u
        cost = jnp.sum(x ** 2) + jnp.sum(u ** 2)
        return -cost * self.dt

    def reset(self, rng: jnp.ndarray) -> State:
        initial_state = jnp.array([jnp.pi, 0.0])
        initial_action = jnp.array([0.0])
        reward, done = self.reward(initial_state, initial_action), jnp.array(0.0)
        pipeline_state = None
        metrics = {}
        return State(pipeline_state=pipeline_state,
                     obs=initial_state,
                     reward=reward,
                     done=done,
                     metrics=metrics)

    def step(self, state: State, action: jnp.ndarray) -> State:
        assert action.shape == (1,)
        x = state.obs
        u = action
        x_next = x + self.dt * self._ode(x, u)
        x_next = x_next.at[0].set(self.convert_angle(x_next[0]))
        reward, done = self.reward(x, u), jnp.array(0.0)
        return state.replace(obs=x_next, reward=reward, done=done)

    @property
    def observation_size(self) -> int:
        return 2

    @property
    def action_size(self) -> int:
        return 1

    @property
    def backend(self) -> str:
        return self._backend


if __name__ == '__main__':
    import jax.random as jr
    from jax import jit

    true_env = Pendulum()


    @jit
    def step(x, u, params):
        env = Pendulum(params=params)
        return env.step(x, u)


    x = true_env.reset(jr.PRNGKey(0))
    u = jnp.array([2.0])

    print(step(x, u, jnp.array([0.1, 5.0])))
    print(step(x, u, jnp.array([1.0, 5.0])))
