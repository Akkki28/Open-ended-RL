import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class HalfCheetahSteadySpeedEnv(MujocoEnv, utils.EzPickle):
    """HalfCheetah variant that rewards steady forward speed.

    Reward = forward_speed * forward_weight - ctrl_cost * ctrl_weight
             - speed_variation * var_weight
    Episodes are truncated after 200 steps.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
    }

    def __init__(
        self,
        xml_file: str = "half_cheetah.xml",
        frame_skip: int = 5,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.1,
        speed_variation_weight: float = 0.5,
        reset_noise_scale: float = 0.1,
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
            speed_variation_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._speed_variation_weight = speed_variation_weight
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )
        # set render_fps after parent init (required by MujocoEnv)
        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))
        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - exclude_current_positions_from_observation
        )
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)
        self.observation_structure = {
            "skipped_qpos": 1 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size - 1 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }
        self._episode_step = 0
        self._prev_velocity = 0.0

    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        observation = self.reset_model()
        self._episode_step = 0
        self._prev_velocity = 0.0
        return observation, {}

    def step(self, action):
        x_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_after = self.data.qpos[0]
        x_velocity = (x_after - x_before) / self.dt

        # reward components
        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self.control_cost(action)
        speed_variation = self._speed_variation_weight * ((x_velocity - self._prev_velocity) ** 2)
        reward = forward_reward - ctrl_cost - speed_variation

        self._prev_velocity = x_velocity
        observation = self._get_obs()
        self._episode_step += 1
        truncated = self._episode_step >= 200
        info = {
            "x_position": x_after,
            "x_velocity": x_velocity,
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_speed_variation": -speed_variation,
        }
        if self.render_mode == "human":
            self.render()
        return observation, float(reward), False, truncated, info

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()
        if self._exclude_current_positions_from_observation:
            position = position[1:]
        return np.concatenate((position, velocity)).ravel()

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def render(self, *args, **kwargs):
        return super().render(*args, **kwargs)

    def close(self):
        super().close()
