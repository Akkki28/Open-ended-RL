import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled


class Continuous_MountainCarDynamicWindCheckpointEnv(gym.Env):
    """Continuous Mountain Car with a time‑varying wind and a single checkpoint.

    Observation: np.array([position, velocity], dtype=np.float32)
    Action: np.array([force], dtype=np.float32) in [-1, 1]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None):
        # ----- static parameters (identical to base env) -----
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.power = 0.0015

        # observation / action spaces
        low_state = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        high_state = np.array([self.max_position, self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_state, high=high_state, dtype=np.float32)
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32)

        # render handling (same as base env)
        self.render_mode = render_mode
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True

        # ----- episode‑specific variables -----
        self.state: np.ndarray = np.zeros(2, dtype=np.float32)
        self.goal_position: float = 0.45
        self.goal_velocity: float = 0.0
        self.wind_amplitude: float = 0.0
        self.wind_period: int = 100
        self.step_counter: int = 0
        self.checkpoint_reached: bool = False

    # ---------------------------------------------------------------------
    # Helper for the hill height (used only for rendering)
    # ---------------------------------------------------------------------
    def _height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55

    # ---------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Sample episode‑specific parameters
        self.goal_position = self.np_random.uniform(0.45, 0.55)
        self.goal_velocity = self.np_random.uniform(0.0, 0.02)
        self.wind_amplitude = self.np_random.uniform(0.0, 0.001)
        self.wind_period = int(self.np_random.integers(50, 151))
        self.step_counter = 0
        self.checkpoint_reached = False

        # Initial state – keep the same range used by the original env
        low, high = utils.maybe_parse_reset_bounds(options, -0.6, -0.4)
        position = self.np_random.uniform(low=low, high=high)
        velocity = 0.0
        self.state = np.array([position, velocity], dtype=np.float32)

        if self.render_mode == "human":
            self.render()
        return self.state.copy(), {}

    # ---------------------------------------------------------------------
    def step(self, action: np.ndarray):
        position, velocity = self.state
        force = float(np.clip(action[0], self.min_action, self.max_action))

        # Time‑varying wind component (adds to the applied force)
        wind = self.wind_amplitude * math.sin(2 * math.pi * self.step_counter / self.wind_period)

        # Update dynamics – same equations as the classic env but with wind
        velocity += (force + wind) * self.power - 0.0025 * math.cos(3 * position)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0.0

        self.state = np.array([position, velocity], dtype=np.float32)

        # ----- termination condition -----
        terminated = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )

        # ----- checkpoint reward (first crossing of -0.2 from left) -----
        checkpoint_reward = 0.0
        if (not self.checkpoint_reached) and (position >= -0.2) and (self.state[0] - velocity < -0.2):
            # The previous position (approx) was left of -0.2, now we are right of it.
            checkpoint_reward = 10.0
            self.checkpoint_reached = True

        # ----- step reward -----
        reward = 0.0
        if terminated:
            reward = 100.0
        reward += checkpoint_reward
        reward -= 0.1 * (action[0] ** 2)  # action magnitude penalty

        # ----- step counter & truncation -----
        self.step_counter += 1
        truncated = False
        if self.step_counter >= 200 and not terminated:
            truncated = True

        if self.render_mode == "human":
            self.render()
        return self.state.copy(), float(reward), terminated, truncated, {}

    # ---------------------------------------------------------------------
    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, e.g. gym.make(\"...\", render_mode=\"rgb_array\")"
            )
            return
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e

        if self.screen is None:
            pygame.display.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.max_position - self.min_position
        scale = self.screen_width / world_width
        carwidth = 40
        carheight = 20
        clearance = 10

        surf = pygame.Surface((self.screen_width, self.screen_height))
        surf.fill((255, 255, 255))

        # draw hill
        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = self._height(xs)
        points = list(zip(((xs - self.min_position) * scale), (ys * scale)))
        pygame.draw.aalines(surf, points, False, (0, 0, 0))

        # car polygon
        pos = self.state[0]
        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        car_coords = []
        for corner in [(l, b), (l, t), (r, t), (r, b)]:
            vec = pygame.math.Vector2(corner).rotate_rad(math.cos(3 * pos))
            car_coords.append(
                (
                    vec[0] + (pos - self.min_position) * scale,
                    vec[1] + clearance + self._height(pos) * scale,
                )
            )
        gfxdraw.aapolygon(surf, car_coords, (0, 0, 0))
        gfxdraw.filled_polygon(surf, car_coords, (0, 0, 0))

        # wheels
        for wheel_offset in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
            vec = pygame.math.Vector2(wheel_offset).rotate_rad(math.cos(3 * pos))
            wheel = (
                int(vec[0] + (pos - self.min_position) * scale),
                int(vec[1] + clearance + self._height(pos) * scale),
            )
            gfxdraw.aacircle(surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128))
            gfxdraw.filled_circle(surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128))

        # goal flag
        flagx = int((self.goal_position - self.min_position) * scale)
        flagy1 = int(self._height(self.goal_position) * scale)
        flagy2 = flagy1 + 50
        gfxdraw.vline(surf, flagx, flagy1, flagy2, (0, 0, 0))
        gfxdraw.aapolygon(
            surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        gfxdraw.filled_polygon(
            surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )

        # checkpoint flag (optional visual aid)
        chk_x = int((-0.2 - self.min_position) * scale)
        chk_y = int(self._height(-0.2) * scale)
        gfxdraw.vline(surf, chk_x, chk_y, chk_y + 30, (0, 128, 0))
        gfxdraw.aacircle(surf, chk_x, chk_y + 30, 5, (0, 128, 0))
        gfxdraw.filled_circle(surf, chk_x, chk_y + 30, 5, (0, 128, 0))

        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    # ---------------------------------------------------------------------
    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
