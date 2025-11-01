# Adapted From: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/hopper_v4.py

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import mujoco
import torch
from env.swimmer import CustomEnvClass

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


class HopperEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 125,
    }

    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-3,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_state_range=(-100.0, 100.0),
        healthy_z_range=(0.7, float("inf")),
        healthy_angle_range=(-0.2, 0.2),
        reset_noise_scale=5e-3,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_state_range,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64
            )

        MujocoEnv.__init__(
            self,
            "hopper.xml",
            4,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        z, angle = self.data.qpos[1:3]
        state = self.state_vector()[2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle

        is_healthy = all((healthy_state, healthy_z, healthy_angle))

        return is_healthy

    @property
    def terminated(self):
        terminated = not self.is_healthy if self._terminate_when_unhealthy else False
        return terminated

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = np.clip(self.data.qvel.flat.copy(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost

        observation = self._get_obs()
        reward = rewards - costs
        terminated = self.terminated
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
    

class HopperBase(HopperEnv, CustomEnvClass):
    F_S = 12
    F_A = 3

    def __init__(
            self,
            forward_reward_weight=1,
            ctrl_cost_weight=0.001,
            healthy_reward=1,
            terminate_when_unhealthy=True,
            healthy_state_range=(-100, 100),
            healthy_z_range=(0.7, float("inf")),
            healthy_angle_range=(-0.2, 0.2),
            reset_noise_scale=0.005,
            exclude_current_positions_from_observation=False,
            trans_model=None,
            **kwargs
    ):
        super().__init__(
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_state_range,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs
        )
        self.n_steps = 0
        self.truncated = False
        self.max_steps = 1e3
        self.trans_model = trans_model
        CustomEnvClass.__init__(self, 'hopper-base')

    def reset_model(self):
        self.state = super().reset_model().astype(np.float32)
        self.n_steps = 0
        self.truncated = False
        return self.state
    
    def _is_healthy(self):
        z, angle = self.state[1:3]
        state = self.state[2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle

        is_healthy = all((healthy_state, healthy_z, healthy_angle))

        return is_healthy
    
    @property
    def is_healthy(self):
        if self.trans_model is not None:
            return self._is_healthy()
        else:
            return super().is_healthy
        
    def get_reward_batch(self, states_list, actions_list):
        # states_list: (BS, T+1, F_S)
        # actions_list: (BS, T, F_A)
        x_velocity = torch.diff(states_list, dim=1)[..., 0] / self.dt
        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self._ctrl_cost_weight * torch.sum(torch.square(actions_list), -1)
        return forward_reward + self.healthy_reward - ctrl_cost  # (BS, T)
    
    def get_reward(self, x_position_before, x_position_after, action):
        x_velocity = (x_position_after - x_position_before) / self.dt
        ctrl_cost = self.control_cost(action)
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward
        rewards = forward_reward + healthy_reward
        costs = ctrl_cost
        return rewards - costs
    
    def _step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        reward = self.get_reward(x_position_before, x_position_after, action)
        observation = self._get_obs().astype(np.float32)
        return reward, observation
   
    def step(self, action):
        assert not self.truncated
        self.state = self.state.astype(np.float32)
        action = action.astype(np.float32)
        if self.trans_model is not None:
            x_position_before = self.state[0]
            next_state = self.trans_model(self.state, action)
            x_position_after = next_state[0]
            reward = self.get_reward(x_position_before, x_position_after, action)
            self.state = next_state
        else:
            reward, self.state = self._step(action)

        self.n_steps += 1
        self.truncated = self.n_steps >= self.max_steps

        if self.render_mode == "human":
            self.render()
        return self.state, reward, self.terminated, self.truncated, {}


class HopperStochWindGaussian(HopperBase):
    def __init__(
            self,
            forward_reward_weight=1,
            ctrl_cost_weight=0.001,
            healthy_reward=1,
            terminate_when_unhealthy=True,
            healthy_state_range=(-100, 100),
            healthy_z_range=(0.7, float("inf")),
            healthy_angle_range=(-0.2, 0.2),
            reset_noise_scale=0.005,
            exclude_current_positions_from_observation=False,
            trans_model=None,
            **kwargs
    ):
        super().__init__(
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_state_range,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            trans_model,
            **kwargs
        )
        CustomEnvClass.__init__(self, 'hopper-gauss-wind')

    def _step_mujoco_simulation(self, ctrl, n_frames):
        self.data.ctrl[:] = ctrl
        # self.model.opt.wind = -np.abs(np.random.randn() * 30)
        for _ in range(n_frames):
            self.model.opt.wind = np.random.randn() * 5
            # self.model.opt.gravity[-1] += np.random.randn(1) * .5
            # self.model.joint('foot_joint').stiffness = np.random.binomial(1, .5) * 1000
            # self.model.opt.density = 1.0
            mujoco.mj_step(self.model, self.data, nstep=1)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)


class POMDPHopperNoPosition(HopperStochWindGaussian):
    F_S = 10
    F_A = 3

    def __init__(
            self,
            forward_reward_weight=1,
            ctrl_cost_weight=0.001,
            healthy_reward=1,
            terminate_when_unhealthy=True,
            healthy_state_range=(-100, 100),
            healthy_z_range=(0.7, float("inf")),
            healthy_angle_range=(-0.2, 0.2),
            reset_noise_scale=0.005,
            exclude_current_positions_from_observation=False,
            trans_model=None,
            **kwargs
    ):
        super().__init__(
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_state_range,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            trans_model,
            **kwargs
        )
        self.env_name = 'hopper-gauss-wind-no-pos'
        self.mask = np.ones(12).astype('bool')
        # Hide positions of torso
        self.mask[:2] = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, rest = super().reset(seed=seed, options=options)
        self.state = obs[self.mask]
        return self.state, rest

    def step(self, action):
        # Position-x, Position-y, [Position-other], [Velocity]
        observation, reward, terminated, self.truncated, info = super().step(action)
        return observation[self.mask], reward, terminated, self.truncated, info


class POMDPHopperNoVelocity(HopperStochWindGaussian):
    F_S = 11
    F_A = 3
    def __init__(
            self,
            forward_reward_weight=1,
            ctrl_cost_weight=0.001,
            healthy_reward=1,
            terminate_when_unhealthy=True,
            healthy_state_range=(-100, 100),
            healthy_z_range=(0.7, float("inf")),
            healthy_angle_range=(-0.2, 0.2),
            reset_noise_scale=0.005,
            exclude_current_positions_from_observation=False,
            trans_model=None,
            **kwargs
    ):
        super().__init__(
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_state_range,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            trans_model,
            **kwargs
        )
        self.env_name = 'hopper-gauss-wind-no-vel'
        self.mask = np.ones(12).astype('bool')
        # Hide angular velocities of torso
        self.mask[-4] = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, rest = super().reset(seed=seed, options=options)
        self.state = obs[self.mask]
        return self.state, rest

    def step(self, action):
        # Position-x, Position-y, [Position-other], [Velocity]
        observation, reward, terminated, self.truncated, info = super().step(action)
        return observation[self.mask], reward, terminated, self.truncated, info


if __name__ == "__main__":
    env = HopperStochWindGaussian(render_mode='human', terminate_when_unhealthy=False)

    import time

    obs, _ = env.reset()
    while True:
        t0 = time.perf_counter()
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        if env.render_mode == "human":
            env.render()
            time.sleep(max(0.0, env.unwrapped.dt - (time.perf_counter() - t0)))
        if term or trunc:
            break
    env.close()



