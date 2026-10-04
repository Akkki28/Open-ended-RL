import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled


class ContinuousMountainCarTimeVaryingEnv(gym.Env):
    """Continuous Mountain Car where the hill shape changes over time.

    The observation space is a 2‑D Box containing position and velocity.
    The action space is a 1‑D Box with continuous force in [-1, 1].
    A small checkpoint at position >= 0.0 yields a bonus reward.
    The hill's slope is modulated by a sinusoidal phase that evolves each step.
    Episodes end after reaching the goal, falling into the left bound, or after 200 steps.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None, goal_velocity: float = 0.0):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45  # same as base env
        self.goal_velocity = goal_velocity
        self.power = 0.0015

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
        self.max_steps = 200
        self.current_step = 0
        self.time = 0.0  # continuous time used for terrain phase
        self.checkpoint_reached = False

    def _terrain_phase(self) -> float:
        """Return a slowly varying phase that perturbs the hill.
        The phase is a sine function of the internal time counter, bounded in [-0.5, 0.5].
        """
        return 0.5 * math.sin(0.05 * self.time)

    def step(self, action: np.ndarray):
        position, velocity = self.state
        # Clip and extract scalar force
        force = float(np.clip(action[0], self.min_action, self.max_action))

        # Time‑varying terrain modifies the cosine term
        phase = self._terrain_phase()
        velocity += force * self.power - 0.0025 * math.cos(3 * position + phase)
        # Clip velocity
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        # Clip position and handle boundary conditions
        if position > self.max_position:
            position = self.max_position
        if position < self.min_position:
            position = self.min_position
        if position == self.min_position and velocity < 0:
            velocity = 0.0

        self.state = np.array([position, velocity], dtype=np.float32)

        # Termination condition (goal reached)
        terminated = bool(position >= self.goal_position and velocity >= self.goal_velocity)

        # Reward calculation
        reward = 0.0
        if terminated:
            reward += 200.0  # large bonus for reaching the goal
        else:
            # checkpoint reward at x >= 0.0 (only once per episode)
            if (not self.checkpoint_reached) and position >= 0.0:
                reward += 10.0
                self.checkpoint_reached = True
        # Small penalty for action magnitude
        reward -= 0.1 * (force ** 2)

        # Step bookkeeping
        self.current_step += 1
        self.time += 1.0  # increment internal time (could be scaled differently)
        truncated = False
        if self.current_step >= self.max_steps and not terminated:
            truncated = True

        if self.render_mode == "human":
            self.render()
        return self.state, float(reward), terminated, truncated, {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        low, high = utils.maybe_parse_reset_bounds(options, -0.6, -0.4)
        init_position = self.np_random.uniform(low=low, high=high)
        self.state = np.array([init_position, 0.0], dtype=np.float32)
        # reset bookkeeping
        self.current_step = 0
        self.time = 0.0
        self.checkpoint_reached = False
        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def _height(self, xs):
        """Base hill height (unchanged from original env)."""
        return np.sin(3 * xs) * 0.45 + 0.55

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
            c_vec = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            coords.append(
                (
                    c_vec[0] + (pos - self.min_position) * scale,
                    c_vec[1] + clearance + self._height(pos) * scale,
                )
            )
        gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))

        for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
            c_vec = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            wheel = (
                int(c_vec[0] + (pos - self.min_position) * scale),
                int(c_vec[1] + clearance + self._height(pos) * scale),
            )
            gfxdraw.aacircle(self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128))
            gfxdraw.filled_circle(self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128))

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

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
