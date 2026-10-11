import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class HalfCheetahSpeedTargetEnv(MujocoEnv, utils.EzPickle):
    """HalfCheetah variant where the agent must match a target forward speed.

    The environment keeps the exact same observation and action spaces as the
    standard HalfCheetah. At reset a target speed v_target is sampled uniformly
    from [2.0, 5.0] m/s. The reward consists of a speed‑matching term and a
    control‑cost term. Episodes are limited to 200 steps (truncated flag).
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
        ctrl_cost_weight: float = 0.1,
        reset_noise_scale: float = 0.1,
        speed_weight: float = 1.0,
        exclude_current_positions_from_observation: bool = True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            ctrl_cost_weight,
            reset_noise_scale,
            speed_weight,
            exclude_current_positions_from_observation,
            **kwargs,
        )
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._speed_weight = speed_weight
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
        # update render_fps after MujocoEnv sets self.dt
        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))
        # observation space mirrors the base HalfCheetah
        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - exclude_current_positions_from_observation
        )
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)
        # episode step counter
        self._step_counter = 0
        # placeholder for target speed
        self._target_speed = 0.0

    # ---------------------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------------------
    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()
        if self._exclude_current_positions_from_observation:
            position = position[1:]
        return np.concatenate((position, velocity)).ravel()

    def _get_rew(self, x_velocity: float, action):
        # Speed‑matching reward (higher when closer to target)
        speed_error = np.abs(x_velocity - self._target_speed)
        speed_reward = -self._speed_weight * speed_error
        ctrl_cost = self.control_cost(action)
        reward = speed_reward - ctrl_cost
        reward_info = {
            "reward_speed": speed_reward,
            "reward_ctrl": -ctrl_cost,
            "target_speed": self._target_speed,
        }
        return reward, reward_info

    # ---------------------------------------------------------------------
    # Gymnasium API
    # ---------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Sample a new target speed for this episode
        self._target_speed = self.np_random.uniform(low=2.0, high=5.0)
        # Reset the MuJoCo state with noise
        observation = self.reset_model()
        self._step_counter = 0
        info = {"target_speed": float(self._target_speed)}
        info.update(self._get_reset_info())
        return observation, info

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        self._step_counter += 1

        terminated = False  # No explicit termination condition
        truncated = self._step_counter >= 200

        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            **reward_info,
        }
        if self.render_mode == "human":
            self.render()
        return observation, float(reward), terminated, truncated, info

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

    def render(self, *args, **kwargs):
        # Delegate to MujocoEnv's render implementation
        return super().render(*args, **kwargs)

    def close(self):
        super().close()
