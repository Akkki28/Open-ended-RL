import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {"distance": 4.0}

class HalfCheetahLaneKeepingEnv(MujocoEnv, utils.EzPickle):
    """HalfCheetah with a lane‑keeping penalty.

    The agent receives a reward composed of forward velocity, control cost, and a
    penalty proportional to the absolute lateral (y‑axis) displacement of the torso.
    Episodes are limited to 200 steps.
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"]}

    def __init__(
        self,
        xml_file: str = "half_cheetah.xml",
        frame_skip: int = 5,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.1,
        lane_penalty_weight: float = 0.5,
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
            lane_penalty_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._lane_penalty_weight = lane_penalty_weight
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
        # Observation space matches the original HalfCheetahEnv
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
        self._step_counter = 0

    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    def lane_penalty(self):
        # qpos[1] is the lateral (y) position of the torso for the HalfCheetah model
        lateral = self.data.qpos[1]
        return self._lane_penalty_weight * np.abs(lateral)

    def _get_rew(self, x_velocity: float, action):
        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self.control_cost(action)
        lane_pen = self.lane_penalty()
        reward = forward_reward - ctrl_cost - lane_pen
        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_lane": -lane_pen,
        }
        return reward, reward_info

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()
        if self._exclude_current_positions_from_observation:
            position = position[1:]
        return np.concatenate((position, velocity)).ravel()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_counter = 0
        observation = self.reset_model()
        info = self._get_reset_info()
        return observation, info

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt
        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        self._step_counter += 1
        terminated = False
        truncated = self._step_counter >= 200
        info = {"x_position": x_position_after, "x_velocity": x_velocity, **reward_info}
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_reset_info(self):
        return {"x_position": self.data.qpos[0]}

    def render(self):
        # Delegate to MujocoEnv's rendering if a mode is set
        if self.render_mode is not None:
            return super().render()

    def close(self):
        return super().close()
