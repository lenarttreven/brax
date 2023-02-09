# Copyright 2022 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint:disable=g-multiple-import
"""Physics pipeline for fully articulated dynamics and collisiion."""

from brax.v2 import actuator
from brax.v2 import geometry
from brax.v2 import kinematics
from brax.v2.base import Motion, System
from brax.v2.spring import collisions
from brax.v2.spring import com
from brax.v2.spring import integrator
from brax.v2.spring import joints
from brax.v2.spring.base import State
from jax import numpy as jp


def init(sys: System, q: jp.ndarray, qd: jp.ndarray) -> State:
  """Initializes physics state.

  Args:
    sys: a brax system
    q: (q_size,) joint angle vector
    qd: (qd_size,) joint velocity vector

  Returns:
    state: initial physics state
  """
  # position/velocity level terms
  x, xd = kinematics.forward(sys, q, qd)
  j, jd, a_p, a_c = kinematics.world_to_joint(sys, x, xd)
  contact = geometry.contact(sys, x)
  x_i, xd_i = com.from_world(sys, x, xd)
  i_inv = com.inv_inertia(sys, x)
  mass = sys.link.inertia.mass ** (1 - sys.spring_mass_scale)

  return State(
      q=q,
      qd=qd,
      x=x,
      xd=xd,
      contact=contact,
      x_i=x_i,
      xd_i=xd_i,
      j=j,
      jd=jd,
      a_p=a_p,
      a_c=a_c,
      i_inv=i_inv,
      mass=mass,
  )


def step(sys: System, state: State, act: jp.ndarray) -> State:
  """Performs a single physics step using spring-based dynamics.

  Resolves actuator forces, joints, and forces at acceleration level, and
  resolves collisions at velocity level with baumgarte stabilization.

  Args:
    sys: system defining the kinematic tree and other properties
    state: physics state prior to step
    act: (act_size,) actuator input vector

  Returns:
    x: updated link transform in world frame
    xd: updated link motion in world frame
  """
  # pre-calculate some auxilliary terms used further down
  state = state.replace(contact=geometry.contact(sys, state.x))
  state = state.replace(i_inv=com.inv_inertia(sys, state.x))

  # calculate acceleration and delta-velocity terms
  tau = actuator.to_tau(sys, act, state.q)
  xdd_i = joints.resolve(sys, state, tau) + Motion.create(vel=sys.gravity)
  # semi-implicit euler: apply acceleration update before resolving collisions
  # TODO: determine whether we really need this extra integration
  state = state.replace(xd_i=state.xd_i + xdd_i * sys.dt)
  xdv_i = collisions.resolve(sys, state)

  # now integrate and update position/velocity-level terms
  x_i, xd_i = integrator.integrate(sys, state.x_i, state.xd_i, xdv_i)
  x, xd = com.to_world(sys, x_i, xd_i)
  state = state.replace(x=x, xd=xd, x_i=x_i, xd_i=xd_i)
  j, jd, a_p, a_c = kinematics.world_to_joint(sys, x, xd)
  q, qd = kinematics.inverse(sys, j, jd)
  state = state.replace(q=q, qd=qd, a_p=a_p, a_c=a_c, j=j, jd=jd)

  return state
