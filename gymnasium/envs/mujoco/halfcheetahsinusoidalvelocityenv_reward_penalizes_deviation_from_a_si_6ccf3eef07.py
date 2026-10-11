import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {"distance": 4.0}

class HalfCheetahSinusoidalVelocityEnv(MujocoEnv, utils.EzPickle):
    """HalfCheetah with a sinusoidal target velocity.

    The agent receives a reward based on how close its forward speed is to a
    sinusoidal reference signal and on control effort.
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"]}

    def __init__(
        self,
        xml_file: str = "half_cheetah.xml",
        frame_skip: int = 5,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        amplitude: float = 2.0,
        period: int = 100,
        ctrl_cost_weight: float = 0.1,
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            amplitude,
            period,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )
        self._amplitude = amplitude
        self._period = period
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )
        # Observation space mirrors the original HalfCheetah environment
        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - exclude_current_positions_from_observation
        )
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)
        self.step_counter = 0

    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.step_counter = 0
        return observation, info

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        # sinusoidal target velocity
        target_vel = self._amplitude * np.sin(2 * np.pi * self.step_counter / self._period)
        speed_error = x_velocity - target_vel
        speed_reward = -np.abs(speed_error)  # penalize absolute error
        ctrl_cost = self.control_cost(action)
        reward = speed_reward - ctrl_cost

        self.step_counter += 1
        truncated = self.step_counter >= 200
        terminated = False

        observation = self._get_obs()
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "target_velocity": target_vel,
            "speed_error": speed_error,
            "reward_speed": speed_reward,
            "reward_ctrl": -ctrl_cost,
        }
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def render(self):
        return super().render()

    def close(self):
        return super().close()
