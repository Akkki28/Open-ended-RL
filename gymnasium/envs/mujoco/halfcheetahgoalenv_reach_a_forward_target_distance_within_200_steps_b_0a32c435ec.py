import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class HalfCheetahGoalEnv(MujocoEnv, utils.EzPickle):
    """HalfCheetah variant with a forward distance goal.

    The agent receives a bonus for approaching a target position (default 10.0 m)
    while still being penalised for control effort. Episodes end after 200 steps
    (truncated) or if the underlying Mujoco termination condition occurs.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
    }

    def __init__(
        self,
        xml_file: str = "half_cheetah.xml",
        frame_skip: int = 5,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.1,
        goal_reward_weight: float = 0.5,
        target_distance: float = 10.0,
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
            goal_reward_weight,
            target_distance,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._goal_reward_weight = goal_reward_weight
        self._target_distance = target_distance
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
        # set render_fps after parent init (MujocoEnv expects it there)
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

    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    def step(self, action):
        # forward progress
        x_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_after = self.data.qpos[0]
        x_vel = (x_after - x_before) / self.dt

        # observation
        observation = self._get_obs()
        # reward components
        forward_reward = self._forward_reward_weight * x_vel
        ctrl_cost = self.control_cost(action)
        # goal reward: negative absolute distance to target (closer -> larger reward)
        distance_error = self._target_distance - x_after
        goal_reward = self._goal_reward_weight * (-abs(distance_error))
        reward = forward_reward - ctrl_cost + goal_reward
        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_goal": goal_reward,
        }
        # step counter handling
        self._episode_step += 1
        truncated = False
        if self._episode_step >= 200:
            truncated = True
        info = {
            "x_position": x_after,
            "x_velocity": x_vel,
            "distance_error": distance_error,
            **reward_info,
        }
        if self.render_mode == "human":
            self.render()
        # HalfCheetah never sets terminated=True by itself
        terminated = False
        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()
        if self._exclude_current_positions_from_observation:
            position = position[1:]
        return np.concatenate((position, velocity)).ravel()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # reset Mujoco state with noise
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        self.set_state(qpos, qvel)
        self._episode_step = 0
        observation = self._get_obs()
        info = {"x_position": self.data.qpos[0]}
        return observation, info

    def render(self):
        return super().render()

    def close(self):
        return super().close()
