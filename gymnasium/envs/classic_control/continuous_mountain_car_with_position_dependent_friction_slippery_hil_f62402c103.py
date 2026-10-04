import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled


class ContinuousMountainCarSlipperyEnv(gym.Env):
    """Continuous Mountain Car where friction varies with position.

    The observation space is a 2‑D vector (position, velocity) and the action
    space is a scalar continuous force in [-1, 1]. The goal is at position 0.45
    with optional goal velocity. A checkpoint at position 0.25 yields a small
    bonus when first crossed.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None, goal_velocity: float = 0.0):
        # ----- environment constants -----
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45
        self.goal_velocity = goal_velocity
        self.power = 0.0015
        # friction modulation: factor in [0.8, 1.2] depending on position
        self.friction_amplitude = 0.2
        self.friction_frequency = 5.0
        # checkpoint
        self.checkpoint_position = 0.25
        self.checkpoint_reward = 10.0
        # state bounds
        self.low_state = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high_state = np.array([self.max_position, self.max_speed], dtype=np.float32)

        self.render_mode = render_mode
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)

        # episode bookkeeping
        self._step_counter = None
        self._checkpoint_reached = None
        self.state = None

    # ---------------------------------------------------------------------
    # Helper for position‑dependent friction
    # ---------------------------------------------------------------------
    def _friction_factor(self, position: float) -> float:
        """Return a multiplier in [1‑amp, 1+amp] that scales the effective acceleration.
        Larger factor -> more resistance.
        """
        return 1.0 + self.friction_amplitude * math.sin(self.friction_frequency * position)

    # ---------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        low, high = utils.maybe_parse_reset_bounds(options, -0.6, -0.4)
        position = self.np_random.uniform(low=low, high=high)
        self.state = np.array([position, 0.0], dtype=np.float32)
        self._step_counter = 0
        self._checkpoint_reached = False
        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    # ---------------------------------------------------------------------
    def step(self, action: np.ndarray):
        position, velocity = self.state
        # clamp action
        force = float(np.clip(action[0], self.min_action, self.max_action))
        # basic acceleration term
        accel = force * self.power - 0.0025 * math.cos(3 * position)
        # apply position‑dependent friction
        accel *= 1.0 / self._friction_factor(position)
        velocity += accel
        # clip speed
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        # enforce bounds
        if position > self.max_position:
            position = self.max_position
        if position < self.min_position:
            position = self.min_position
        if position == self.min_position and velocity < 0:
            velocity = 0.0

        self.state = np.array([position, velocity], dtype=np.float32)
        self._step_counter += 1

        # termination condition
        terminated = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        # truncation due to step limit
        truncated = False
        if self._step_counter >= 200 and not terminated:
            truncated = True

        # reward shaping
        reward = -0.1 * (force ** 2)  # action penalty
        reward -= 0.01  # small step penalty to encourage speed
        if terminated:
            reward += 100.0
        # checkpoint reward (first time crossing)
        if (not self._checkpoint_reached) and (position >= self.checkpoint_position):
            reward += self.checkpoint_reward
            self._checkpoint_reached = True

        if self.render_mode == "human":
            self.render()
        info = {}
        return np.array(self.state, dtype=np.float32), float(reward), terminated, truncated, info

    # ---------------------------------------------------------------------
    def _height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55

    # ---------------------------------------------------------------------
    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
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
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))
        pos = self.state[0]
        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = self._height(xs)
        xys = list(zip(((xs - self.min_position) * scale), (ys * scale), strict=True))
        pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))
        clearance = 10
        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            coords.append(
                (
                    c[0] + (pos - self.min_position) * scale,
                    c[1] + clearance + self._height(pos) * scale,
                )
            )
        gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))
        for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            wheel = (
                int(c[0] + (pos - self.min_position) * scale),
                int(c[1] + clearance + self._height(pos) * scale),
            )
            gfxdraw.aacircle(self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128))
            gfxdraw.filled_circle(self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128))
        # flag for goal
        flagx = int((self.goal_position - self.min_position) * scale)
        flagy1 = int(self._height(self.goal_position) * scale)
        flagy2 = flagy1 + 50
        gfxdraw.vline(self.surf, flagx, flagy1, flagy2, (0, 0, 0))
        gfxdraw.aapolygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        gfxdraw.filled_polygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        # checkpoint flag
        chkx = int((self.checkpoint_position - self.min_position) * scale)
        chky1 = int(self._height(self.checkpoint_position) * scale)
        chky2 = chky1 + 30
        gfxdraw.vline(self.surf, chkx, chky1, chky2, (0, 128, 0))
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
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
