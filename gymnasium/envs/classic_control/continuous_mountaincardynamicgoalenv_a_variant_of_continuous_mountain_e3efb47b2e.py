import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled


class Continuous_MountainCarDynamicGoalEnv(gym.Env):
    """Continuous Mountain Car with a randomly moving goal and stochastic wind.

    The observation space is a 2‑D Box: [position, velocity].
    The action space is a 1‑D continuous force in [-1, 1].
    The episode length is capped at 200 steps.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None):
        # ----- environment constants (same as original) -----
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.power = 0.0015
        # ----- dynamic goal ranges -----
        self.goal_position_range = (0.40, 0.50)  # inclusive range for the target
        self.goal_velocity_range = (0.0, 0.01)
        # ----- wind strength -----
        self.wind_strength = 0.001  # uniform noise added to force each step

        # observation and action spaces (identical to base env)
        low_state = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        high_state = np.array([self.max_position, self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_state, high=high_state, dtype=np.float32)
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32)

        self.render_mode = render_mode
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True

        # episode bookkeeping
        self.state = None
        self.goal_position = None
        self.goal_velocity = None
        self.step_counter = None
        self.np_random = None

    # ---------------------------------------------------------------------
    # Helper: mountain height for rendering
    # ---------------------------------------------------------------------
    def _height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55

    # ---------------------------------------------------------------------
    # Reset – samples a new goal, resets position, velocity and step counter
    # ---------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Sample goal position and velocity uniformly within the defined ranges
        self.goal_position = self.np_random.uniform(*self.goal_position_range)
        self.goal_velocity = self.np_random.uniform(*self.goal_velocity_range)
        # Sample initial position (same bounds as original env) and zero velocity
        low, high = utils.maybe_parse_reset_bounds(options, -0.6, -0.4)
        init_pos = self.np_random.uniform(low=low, high=high)
        self.state = np.array([init_pos, 0.0], dtype=np.float32)
        self.step_counter = 0
        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    # ---------------------------------------------------------------------
    # Step – applies action, adds wind noise, updates dynamics, computes reward
    # ---------------------------------------------------------------------
    def step(self, action: np.ndarray):
        position, velocity = self.state
        # Clip action to allowed range
        force = float(np.clip(action[0], self.min_action, self.max_action))
        # Add wind noise (zero‑mean uniform)
        wind = self.np_random.uniform(-self.wind_strength, self.wind_strength)
        total_force = force + wind
        # Physics (identical to original, just using total_force)
        velocity += total_force * self.power - 0.0025 * math.cos(3 * position)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        # Enforce position bounds and zero‑velocity bounce at left wall
        if position > self.max_position:
            position = self.max_position
        if position < self.min_position:
            position = self.min_position
            if velocity < 0:
                velocity = 0.0
        self.state = np.array([position, velocity], dtype=np.float32)

        # Termination condition based on the sampled goal
        terminated = bool(position >= self.goal_position and velocity >= self.goal_velocity)

        # Reward shaping:
        #   +100 for reaching the goal
        #   -0.1 * action^2 (fuel penalty, same magnitude as original)
        #   -0.01 per time step to encourage faster solutions
        reward = -0.01  # time penalty
        reward -= 0.1 * (action[0] ** 2)
        if terminated:
            reward += 100.0

        # Step counter and truncation handling (200‑step horizon)
        self.step_counter += 1
        truncated = False
        if self.step_counter >= 200 and not terminated:
            truncated = True

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), float(reward), terminated, truncated, {}

    # ---------------------------------------------------------------------
    # Rendering – unchanged from the original implementation
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
            else:  # rgb_array
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        world_width = self.max_position - self.min_position
        scale = self.screen_width / world_width
        carwidth = 40
        carheight = 20
        # Background
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))
        # Track
        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = self._height(xs)
        xys = list(zip(((xs - self.min_position) * scale), (ys * scale), strict=True))
        pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))
        # Car
        pos = self.state[0]
        clearance = 10
        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            vec = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            coords.append(
                (
                    vec[0] + (pos - self.min_position) * scale,
                    vec[1] + clearance + self._height(pos) * scale,
                )
            )
        gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))
        # Wheels
        for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
            vec = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            wheel = (
                int(vec[0] + (pos - self.min_position) * scale),
                int(vec[1] + clearance + self._height(pos) * scale),
            )
            gfxdraw.aacircle(self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128))
            gfxdraw.filled_circle(self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128))
        # Goal flag (use the sampled goal_position)
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
        # Flip and blit
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    # ---------------------------------------------------------------------
    # Close – clean up pygame resources
    # ---------------------------------------------------------------------
    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
