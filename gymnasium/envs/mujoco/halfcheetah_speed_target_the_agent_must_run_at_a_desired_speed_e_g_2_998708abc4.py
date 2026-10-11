import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {"distance": 4.0}

class HalfCheetahSpeedTargetEnv(MujocoEnv, utils.EzPickle):
    """HalfCheetah variant that rewards maintaining a target forward speed.

    The observation and action spaces are identical to the standard HalfCheetah
    environment.  Episodes are capped at 200 steps inside the environment.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
        "render_fps": None,
    }

    def __init__(
        self,
        xml_file: str = "half_cheetah.xml",
        frame_skip: int = 5,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.1,
        reset_noise_scale: float = 0.1,
        target_speed: float = 2.0,
        exclude_current_positions_from_observation: bool = True,
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
            target_speed,
            exclude_current_positions_from_observation,
            **kwargs,
        )
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._target_speed = target_speed
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )
        # Update render_fps based on simulation dt
        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))
        # Observation space definition matches the base environment
        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - exclude_current_positions_from_observation
        )
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)
        # internal step counter for truncation
        self._episode_step = 0

    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_step = 0
        observation = self.reset_model()
        info = self._get_reset_info()
        return observation, info

    def step(self, action):
        x_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_after = self.data.qpos[0]
        x_vel = (x_after - x_before) / self.dt
        observation = self._get_obs()
        reward, reward_info = self._compute_reward(x_vel, action)
        self._episode_step += 1
        truncated = self._episode_step >= 200
        terminated = False  # no explicit termination condition
        info = {"x_position": x_after, "x_velocity": x_vel, **reward_info}
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def _compute_reward(self, x_velocity, action):
        # Penalise deviation from target speed and control effort
        speed_error = np.abs(x_velocity - self._target_speed)
        speed_reward = -self._forward_reward_weight * speed_error
        ctrl_cost = self.control_cost(action)
        reward = speed_reward - ctrl_cost
        reward_info = {
            "reward_speed": speed_reward,
            "reward_ctrl": -ctrl_cost,
            "target_speed": self._target_speed,
        }
        return reward, reward_info

    def render(self, *args, **kwargs):
        return super().render(*args, **kwargs)

    def close(self):
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.close()
            self.viewer = None
