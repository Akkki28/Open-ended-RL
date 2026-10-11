import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {"distance": 4.0}


class HalfCheetahDirectionEnv(MujocoEnv, utils.EzPickle):
    """HalfCheetah environment where the goal direction (+1 forward, -1 backward) is sampled each episode.
    Reward = direction * forward_speed - ctrl_cost.
    Episode ends after 200 steps (truncated) or if the cheetah falls excessively.
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"]}

    def __init__(
        self,
        xml_file: str = "half_cheetah.xml",
        frame_skip: int = 5,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
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
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )
        self._forward_reward_weight = forward_reward_weight
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
        # Set render_fps after parent init to avoid validation errors
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
        # episode bookkeeping
        self._step_counter = 0
        self._target_dir = 1.0

    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    def reset(self, seed=None, options=None):
        # Reset Mujoco state
        observation, info = super().reset(seed=seed, options=options)
        # Sample new target direction (+1 or -1)
        self._target_dir = self.np_random.choice(np.array([1.0, -1.0]))
        self._step_counter = 0
        info = {"target_direction": self._target_dir, **info}
        return observation, info

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        # Reward encourages moving in the sampled direction
        forward_reward = self._forward_reward_weight * self._target_dir * x_velocity
        ctrl_cost = self.control_cost(action)
        reward = forward_reward - ctrl_cost

        self._step_counter += 1
        terminated = False
        truncated = self._step_counter >= 200

        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "target_direction": self._target_dir,
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }
        if self.render_mode == "human":
            self.render()
        return observation, float(reward), terminated, truncated, info

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
        return super().close()
