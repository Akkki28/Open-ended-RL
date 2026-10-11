import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class HalfCheetahSinusoidalVelocityEnv(MujocoEnv, utils.EzPickle):
    """HalfCheetah tracking a sinusoidal forward velocity.

    The target velocity is v_target(t) = A * sin(2π f t) where
    A=2.0 (m/s) and f=0.05 Hz.  Reward = -|v - v_target| - ctrl_cost.
    Episodes are truncated after 200 steps.
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"]}

    def __init__(
        self,
        xml_file: str = "half_cheetah.xml",
        frame_skip: int = 5,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        ctrl_cost_weight: float = 0.1,
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        amplitude: float = 2.0,
        frequency: float = 0.05,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            amplitude,
            frequency,
            **kwargs,
        )
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        self._amplitude = amplitude
        self._frequency = frequency
        self._step_counter = 0
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )
        # add render_fps after parent init (MujocoEnv expects it)
        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))
        # observation space identical to base HalfCheetah
        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - exclude_current_positions_from_observation
        )
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)

    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    def step(self, action):
        x_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_after = self.data.qpos[0]
        x_vel = (x_after - x_before) / self.dt
        # time in seconds since episode start
        t = self._step_counter * self.dt
        v_target = self._amplitude * np.sin(2 * np.pi * self._frequency * t)
        # reward components
        vel_error = np.abs(x_vel - v_target)
        ctrl_cost = self.control_cost(action)
        reward = -vel_error - ctrl_cost
        reward_info = {
            "reward_vel_error": -vel_error,
            "reward_ctrl": -ctrl_cost,
            "target_velocity": v_target,
        }
        self._step_counter += 1
        truncated = self._step_counter >= 200
        observation = self._get_obs()
        info = {"x_position": x_after, "x_velocity": x_vel, **reward_info}
        if self.render_mode == "human":
            self.render()
        return observation, float(reward), False, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        observation = self.reset_model()
        self._step_counter = 0
        return observation, self._get_reset_info()

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

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()
        if self._exclude_current_positions_from_observation:
            position = position[1:]
        return np.concatenate((position, velocity)).ravel()

    def render(self):
        return super().render()

    def close(self):
        return super().close()
