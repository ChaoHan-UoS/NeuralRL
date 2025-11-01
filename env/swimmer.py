# Adapted From: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/swimmer_v4.py

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

import torch
from abc import ABC, abstractmethod

import mujoco


class SwimmerEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-4,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        trans_model=None,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64
            )
        MujocoEnv.__init__(
            self, "swimmer.xml", 4, observation_space=observation_space, **kwargs
        )

        self.trans_model = trans_model

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        xy_position_before = self.data.qpos[0:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.qpos[0:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        forward_reward = self._forward_reward_weight * x_velocity

        ctrl_cost = self.control_cost(action)

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        info = {
            "reward_fwd": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, False, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observation = np.concatenate([position, velocity]).ravel()
        return observation

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


class CustomEnvClass(ABC):
    def __init__(self, env_name):
        self.env_name = env_name

    def get_reward_batch(self, states_list: torch.Tensor, actions_list: torch.Tensor):
        # states_list/action_list: B, T, F_s/F_a
        x_velocity = torch.diff(states_list, dim=1)[..., 0] / self.dt
        ctrl_cost = self._ctrl_cost_weight * torch.sum(torch.square(actions_list), -1)
        forward_reward = self._forward_reward_weight * x_velocity
        return forward_reward + self.healthy_reward - ctrl_cost
    
    def get_env_name(self):
        return self.env_name


class SwimmerBase(SwimmerEnv, CustomEnvClass):
    F_S = 10
    F_A = 2

    def __init__(
            self,
            forward_reward_weight=1,
            ctrl_cost_weight=0.0001,
            reset_noise_scale=0.1,
            max_steps=1000,  # Truncation
            exclude_current_positions_from_observation=False,
            trans_model=None,
            **kwargs
    ):
        super().__init__(
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation=exclude_current_positions_from_observation,
            trans_model=trans_model,
            **kwargs
        )
    
        # self.model.opt.integrator = 0
        self.n_steps = 0
        self.truncated = False
        self.max_steps = max_steps
        self.state = None
        # Equivalent to super(utils.EzPickle, self).__init__('base_swimmer')
        # as CustomEnvClass is the next class to EzPickle in the MRO
        CustomEnvClass.__init__(self, 'swimmer-base')

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict | None = None
    ):
        self.n_steps = 0
        self.truncated = False
        to_return = super().reset(seed=seed, options=options)
        self.state = to_return[0].astype(np.float32)
        return self.state, to_return[-1]

    def get_reward(self, xy_position_before, xy_position_after, action):
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity
        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self.control_cost(action)
        reward = forward_reward - ctrl_cost
        return reward
    
    def step(self, action):
        assert not self.truncated
        action = action.astype(np.float32)
        if self.trans_model is not None:
            xy_position_before = self.state[0:2]
            next_state = self.trans_model(self.state, action)
            xy_position_after = next_state[0:2]

            reward = self.get_reward(
                xy_position_before=xy_position_before, xy_position_after=xy_position_after, action=action
            )
            observation = next_state
        else:
            reward, observation = self._step(action)

        if self.render_mode == "human":
            self.render()

        self.n_steps += 1
        self.truncated = self.n_steps >= self.max_steps
        self.state = observation
        return observation, reward, False, self.truncated, {}

    def _step(self, action):
        xy_position_before = self.data.qpos[0:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.qpos[0:2].copy()

        reward = self.get_reward(xy_position_before, xy_position_after, action)
        observation = self._get_obs()
        observation = observation.astype(np.float32)
        return reward, observation
    
    def get_reward_batch(self, states_list, actions_list):
        # states_list: (BS, T+1, F_S)
        # actions_list: (BS, T, F_A)
        x_velocity = torch.diff(states_list, dim=1)[..., 0] / self.dt
        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self._ctrl_cost_weight * torch.sum(torch.square(actions_list), -1)
        return forward_reward - ctrl_cost  # (BS, T)


class SwimmerStochStiffnessZeroMean(SwimmerBase):
    def __init__(
            self,
            forward_reward_weight=1,
            ctrl_cost_weight=0.0001,
            reset_noise_scale=0.1,
            max_steps=1000,
            exclude_current_positions_from_observation=False,
            trans_model=None, std=100, **kwargs
    ):
        super().__init__(
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            max_steps,
            exclude_current_positions_from_observation,
            trans_model,
            **kwargs
        )
        self.std = std
        CustomEnvClass.__init__(self, 'swimmer-stiff')

    def _step_mujoco_simulation(self, ctrl, n_frames):
        self.data.ctrl[:] = ctrl

        for _ in range(n_frames):
            # Introduce randomness to the stiffness of the first actuated joint (joint index 3),
            # deault stiffness is 0
            self.model.joint(3).stiffness = np.random.randn() * self.std
            mujoco.mj_step(self.model, self.data, nstep=1)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)


class POMDPSwimmerNoPosition(SwimmerStochStiffnessZeroMean):
    F_S = 7
    F_A = 2
    def __init__(
            self,
            forward_reward_weight=1,
            ctrl_cost_weight=0.0001,
            reset_noise_scale=0.1,
            max_steps=1000,
            exclude_current_positions_from_observation=False,
            trans_model=None, std=500, **kwargs
    ):
        super().__init__(
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            max_steps,
            exclude_current_positions_from_observation,
            trans_model, std, **kwargs
        )
        self.mask = np.ones(10).astype('bool')
        # Hide positions and angle of the tip
        self.mask[:3] = False
        self.env_name = 'swimmer-stiff-no-pos'

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict | None = None
    ):
        obs, rest = super().reset(seed=seed, options=options)
        self.state = obs[self.mask]
        return self.state, rest

    def step(self, action):
        # Position-x, Position-y, [Position-other], [Velocity]
        observation, reward, terminated, self.truncated, info = super().step(action)
        return observation[self.mask], reward, terminated, self.truncated, info


class POMDPSwimmerNoVelocity(SwimmerStochStiffnessZeroMean):
    F_S = 7
    F_A = 2
    def __init__(
            self,
            forward_reward_weight=1,
            ctrl_cost_weight=0.0001,
            reset_noise_scale=0.1,
            max_steps=1000,
            exclude_current_positions_from_observation=False,
            trans_model=None, std=500, **kwargs
    ):
        super().__init__(
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale, max_steps,
            exclude_current_positions_from_observation,
            trans_model, std, **kwargs
        )
        self.mask = np.ones(10).astype('bool')
        # Hide positional and angular velocities of the tip
        self.mask[5:8] = False
        self.env_name = 'swimmer-stiff-no-vel'

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict | None = None
    ):
        obs, rest = super().reset(seed=seed, options=options)
        self.state = obs[self.mask]
        return self.state, rest

    def step(self, action):
        # Position-x, Position-y, [Position-other], [Velocity]
        observation, reward, terminated, self.truncated, info = super().step(action)
        return observation[self.mask], reward, terminated, self.truncated, info


if __name__ == '__main__':
    # env = POMDPSwimmerNoVelocity(render_mode='human', std=500)
    env = SwimmerStochStiffnessZeroMean(render_mode='human', std=500)

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