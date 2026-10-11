import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class HalfCheetahEnergyEfficiencyEnv(MujocoEnv, utils.EzPickle):
    """HalfCheetah environment that rewards forward speed while penalizing kinetic
    energy and encourages staying within a target speed band.
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
        energy_cost_weight: float = 0.05,
        speed_target_min: float = 0.5,
        speed_target_max: float = 2.0,
        speed_target_bonus: float = 0.5,
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
            energy_cost_weight,
            speed_target_min,
            speed_target_max,
            speed_target_bonus,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._energy_cost_weight = energy_cost_weight
        self._speed_target_min = speed_target_min
        self._speed_target_max = speed_target_max
        self._speed_target_bonus = speed_target_bonus
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
        # set render fps after parent init (MujocoEnv expects it here)
        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))
        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - exclude_current_positions_from_observation
        )
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)
        self._episode_steps = 0

    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    def kinetic_energy_cost(self):
        # simple quadratic kinetic energy (mass omitted, constant factor absorbed)
        return self._energy_cost_weight * np.sum(np.square(self.data.qvel))

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._compute_reward(x_velocity, action)
        info = {"x_position": x_position_after, "x_velocity": x_velocity, **reward_info}

        self._episode_steps += 1
        terminated = False
        truncated = self._episode_steps >= 200

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def _compute_reward(self, x_velocity: float, action):
        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self.control_cost(action)
        energy_cost = self.kinetic_energy_cost()
        # Bonus for staying in the desired speed band
        if self._speed_target_min <= x_velocity <= self._speed_target_max:
            speed_bonus = self._speed_target_bonus
        else:
            speed_bonus = 0.0
        reward = forward_reward - ctrl_cost - energy_cost + speed_bonus
        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_energy": -energy_cost,
            "reward_speed_bonus": speed_bonus,
        }
        return reward, reward_info

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()
        if self._exclude_current_positions_from_observation:
            position = position[1:]
        return np.concatenate((position, velocity)).ravel()

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed)
        self._episode_steps = 0
        return observation, info

    def render(self, *args, **kwargs):
        return super().render(*args, **kwargs)

    def close(self):
        return super().close()
