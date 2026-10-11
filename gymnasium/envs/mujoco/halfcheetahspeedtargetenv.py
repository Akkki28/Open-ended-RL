import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {"distance": 4.0}


class HalfCheetahSpeedTargetEnv(MujocoEnv, utils.EzPickle):
    """HalfCheetah variant where each episode has a random target speed.

    The agent receives a reward based on how close its forward velocity is to the
    target speed while still being penalized for control effort. Episodes are
    limited to 200 steps internally (in addition to any external TimeLimit).
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
        "render_fps": None,
    }

    def __init__(
        self,
        xml_file: str = "half_cheetah.xml",
        frame_skip: int = 5,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 0.0,  # not used directly
        ctrl_cost_weight: float = 0.1,
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        target_speed_range: tuple[float, float] = (0.0, 3.0),
        max_episode_steps: int = 200,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            target_speed_range,
            max_episode_steps,
            **kwargs,
        )
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._target_speed_range = target_speed_range
        self._max_episode_steps = max_episode_steps
        self._episode_step = 0
        self._target_speed = 0.0

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        # Update render fps metadata now that dt is known
        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - exclude_current_positions_from_observation
        )
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)

    # ---------------------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------------------
    def control_cost(self, action: np.ndarray) -> float:
        return self._ctrl_cost_weight * np.sum(np.square(action))

    def _get_obs(self) -> np.ndarray:
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()
        if self._exclude_current_positions_from_observation:
            position = position[1:]
        return np.concatenate((position, velocity)).ravel()

    def _get_rew(self, x_velocity: float, action: np.ndarray) -> tuple[float, dict]:
        # Reward encourages matching the target speed.
        speed_error = x_velocity - self._target_speed
        speed_reward = -np.square(speed_error)  # higher (less negative) when close
        ctrl_cost = self.control_cost(action)
        reward = speed_reward - ctrl_cost
        info = {
            "target_speed": self._target_speed,
            "x_velocity": x_velocity,
            "reward_speed_error": speed_reward,
            "reward_ctrl": -ctrl_cost,
        }
        return reward, info

    # ---------------------------------------------------------------------
    # Gymnasium API
    # ---------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._episode_step = 0
        low, high = self._target_speed_range
        self._target_speed = self.np_random.uniform(low, high)
        observation = self.reset_model()
        info = {"target_speed": self._target_speed, **self._get_reset_info()}
        return observation, info

    def step(self, action: np.ndarray):
        x_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_after = self.data.qpos[0]
        x_vel = (x_after - x_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_vel, action)

        self._episode_step += 1
        terminated = False  # no natural termination condition
        truncated = self._episode_step >= self._max_episode_steps
        info = {"x_position": x_after, "x_velocity": x_vel, **reward_info}

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def render(self):
        # Delegates to the parent class rendering implementation.
        return super().render()

    def close(self):
        super().close()

    # ---------------------------------------------------------------------
    # Reset model & info helpers (same as base HalfCheetah)
    # ---------------------------------------------------------------------
    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_reset_info(self):
        return {"x_position": self.data.qpos[0]}
