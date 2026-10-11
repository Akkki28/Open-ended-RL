import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {"distance": 4.0}


class HalfCheetahVelocityTrackingEnv(MujocoEnv, utils.EzPickle):
    """HalfCheetah variant where the agent must track a target forward speed.

    The observation and action spaces are identical to the standard HalfCheetah
    environment.  At each episode reset a target velocity is sampled uniformly
    from [0.5, 2.0] m/s.  The reward is negative absolute error to the target plus
    a control‑cost penalty.  Episodes are truncated after 200 steps (defensive
    fallback – the outer TimeLimit wrapper also enforces a horizon).
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
        target_velocity_range: tuple[float, float] = (0.5, 2.0),
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
            target_velocity_range,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._target_range = target_velocity_range
        self._step_counter = 0
        self._target_velocity = 0.0

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        # observation space is built exactly as in the reference env
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

        # render fps is derived from the simulation dt
        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------
    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    def _sample_target_velocity(self):
        low, high = self._target_range
        return self.np_random.uniform(low, high)

    # ---------------------------------------------------------------------
    # Gymnasium API
    # ---------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # MujocoEnv.reset calls reset_model internally and returns (obs, info)
        observation, info = self._get_obs(), {}
        self._step_counter = 0
        self._target_velocity = self._sample_target_velocity()
        info["target_velocity"] = float(self._target_velocity)
        # also expose the initial x position for consistency with the base env
        info["x_position"] = float(self.data.qpos[0])
        return observation, info

    def step(self, action):
        x_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_after = self.data.qpos[0]
        x_vel = (x_after - x_before) / self.dt

        self._step_counter += 1
        truncated = self._step_counter >= 200

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_vel, action)
        info = {
            "x_position": float(x_after),
            "x_velocity": float(x_vel),
            "target_velocity": float(self._target_velocity),
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, False, truncated, info

    def _get_rew(self, x_velocity: float, action):
        # tracking reward: negative absolute error to the target
        speed_error = np.abs(x_velocity - self._target_velocity)
        tracking_reward = -self._forward_reward_weight * speed_error
        ctrl_cost = self.control_cost(action)
        reward = tracking_reward - ctrl_cost
        reward_info = {
            "reward_tracking": tracking_reward,
            "reward_ctrl": -ctrl_cost,
        }
        return reward, reward_info

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
