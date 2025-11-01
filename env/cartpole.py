# Adapted from: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

import math
from typing import Optional, Union
import torch
import numpy as np
import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled


class CartPoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    F_S = 4
    F_A = 1

    def __init__(
        self,
        trans_model=None,
        verbose=True,
        sutton_barto_reward: bool = False,
        render_mode: Optional[str] = None,
        force_mag: float = 10.0,
        discrete=True,
    ):
        self._sutton_barto_reward = sutton_barto_reward

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = force_mag
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.discrete = discrete
        self.action_space = (
            spaces.Discrete(2)
            if discrete
            else spaces.Box(-1, +1, (1,), dtype=np.float32)
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

        # Learned Environment
        if verbose:
            if trans_model is not None:
                print("Using a learned model")
                # assert std_value is not None
                # assert mean_value is not None
            else:
                print("Using the original implementation")

        self.trans_model = trans_model

    def __repr__(self) -> str:
        return 'det-cartpole'

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        # assert self.observation_space.contains(
        #     self.state
        # ), f"{self.state!r} ({type(self.state)}) invalid"
        
        if self.trans_model is not None:
            if not isinstance(action, np.ndarray):
                action = np.array([action], dtype=np.float32)
            x, x_dot, theta, theta_dot = self.trans_model(self.state, action)
            info = {}
        else:
            x, x_dot, theta, theta_dot = self.state
            x, x_dot, theta, theta_dot, info = self._next_step(
                action, x, x_dot, theta, theta_dot
            )

        self.state = np.array([x, x_dot, theta, theta_dot])

        terminated = self.is_terminated(self.state)

        if not terminated:
            if self._sutton_barto_reward:
                reward = 0.0
            elif not self._sutton_barto_reward:
                reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            if self._sutton_barto_reward:
                reward = -1.0
            elif not self._sutton_barto_reward:
                reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            if self._sutton_barto_reward:
                reward = -1.0
            elif not self._sutton_barto_reward:
                reward = 0.0

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`

        # Include number of steps. Set truncated to true if the number of steps exceeds 500
        self.steps += 1
        truncation = self.steps >= 500

        return (
            np.array(self.state, dtype=np.float32),
            reward,
            terminated,
            truncation,
            info,
        )

    def is_terminated(self, state):
        x, _, theta, _ = state
        return bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

    def get_force(self, action):
        # Return force to be applied
        if self.discrete:
            force = self.force_mag if action == 1 else -self.force_mag
        else:
            pwr = np.clip(np.abs(action), 0.5, 1).item()
            dir = -1 if action < 0 else 1
            force = self.force_mag * dir * pwr
        return force
    
    def _next_step(self, action, x, x_dot, theta, theta_dot):
        force = self.get_force(action)

        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        info = {}

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        return x, x_dot, theta, theta_dot, info
    
    @property
    def dt(self):
        return self.tau

    def reset(
        self,
        *,
        state=None,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        if state is not None:
            if isinstance(state, torch.Tensor):
                state = state.numpy(force=True)
            self.state = state.astype(np.float32)
        else:
            low, high = utils.maybe_parse_reset_bounds(
                options,
                -0.05,
                0.05,  # default low
            )  # default high
            self.state = self.np_random.uniform(low=low, high=high, size=(4,)).astype(dtype=np.float32)
        self.steps_beyond_terminated = None
        self.steps = 0
        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class CartPoleEnvStochCon(CartPoleEnv):
    F_S = 4
    F_A = 1
    
    def __init__(
        self,
        trans_model=None,
        verbose=False,
        sutton_barto_reward: bool = False,
        render_mode: str | None = None,
        force_mag=20,
        diff_mu=0,
        diff_sigma=5,
    ):
        super().__init__(
            trans_model, verbose, sutton_barto_reward, render_mode, force_mag, discrete=False
        )
        self.diff_mu = diff_mu
        self.diff_sigma = diff_sigma

    def __repr__(self) -> str:
        return "stoch-cartpole"

    def _next_step(self, action, x, x_dot, theta, theta_dot):
        pwr = np.clip(np.abs(action), 0.5, 1).item()
        dir = -1 if action < 0 else 1
        force = self.force_mag * dir * pwr

        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        info = {}

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Add standard Gaussian white noise force in continuous time,
        # i.e., the derivative (in the distribution sense) of a Brownian motion
        # Formulated as a SDE of cart velocity: dx_t' = F_t/m * dt + sigma/m * dW_t
        exter_force_dwt = self.diff_mu + self.diff_sigma * np.random.randn()
        xacc += exter_force_dwt / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        return x, x_dot, theta, theta_dot, info


class CartPoleEnvVaryingLength(CartPoleEnv):
    def __init__(
            self,
            trans_model=None,
            verbose=False,
            sutton_barto_reward: bool = False,
            render_mode: str | None = None,
            discrete=False,
            pole_len=1.4
    ):
        super().__init__(
            trans_model, verbose, sutton_barto_reward, render_mode, discrete=discrete
        )
        self.length = pole_len / 2

    def __repr__(self) -> str:
        return "det-cartpole-varlen"


class CartPoleEnvStochVaryingLength(CartPoleEnvStochCon):
    def __init__(
            self,
            trans_model=None,
            verbose=False,
            sutton_barto_reward: bool = False,
            render_mode: str | None = None,
            force_mag=20,
            diff_mu=0,
            diff_sigma=5,
            pole_len=1.0
    ):
        super().__init__(
            trans_model, verbose, sutton_barto_reward, render_mode, force_mag, diff_mu, diff_sigma
        )
        self.length = pole_len / 2

    def __repr__(self) -> str:
        return "stoch-cartpole-varlen"


if __name__ == "__main__":
    env = CartPoleEnv(render_mode="human", discrete=False)

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
