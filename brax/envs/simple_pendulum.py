from jax import numpy as jnp

from brax.envs.base import State, Env


class Pendulum(Env):

    def __init__(self,
                 backend: str = 'string', ):
        self.dt = 0.1
        self.T_horizon = 10
        self._backend = backend

    def _ode(self, x, u):
        assert x.shape == (2,) and u.shape == (1,)
        system_params = jnp.array([5.0, 9.81])
        u = 4 * u
        return jnp.array([x[1], system_params[1] / system_params[0] * jnp.sin(x[0]) + u.reshape()])

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
