import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class HalfCheetahGoalEnv(MujocoEnv, utils.EzPickle):
    """HalfCheetah variant where the agent must reach a forward target.

    The episode ends after 200 environment steps or when the target is reached.
    Reward = forward_speed - ctrl_cost - distance_error + goal_bonus.
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
        target_range: tuple[float, float] = (5.0, 15.0),
        goal_bonus: float = 10.0,
        max_episode_steps: int = 200,
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
            target_range,
            goal_bonus,
            max_episode_steps,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._target_range = target_range
        self._goal_bonus = goal_bonus
        self._max_episode_steps = max_episode_steps
        self._step_counter = 0
        self._target_distance = 0.0

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
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # reset Mujoco state
        observation = self.reset_model()
        # sample new target distance uniformly within range
        low, high = self._target_range
        self._target_distance = self.np_random.uniform(low, high)
        self._step_counter = 0
        info = {"target_distance": float(self._target_distance)}
        return observation, info

    def step(self, action):
        x_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_after = self.data.qpos[0]
        x_vel = (x_after - x_before) / self.dt

        self._step_counter += 1
        truncated = self._step_counter >= self._max_episode_steps
        # goal reached flag (does not terminate, just gives bonus)
        goal_reached = x_after >= self._target_distance

        observation = self._get_obs()
        reward, reward_info = self._compute_reward(x_vel, action, x_after, goal_reached)
        info = {
            "x_position": float(x_after),
            "x_velocity": float(x_vel),
            "target_distance": float(self._target_distance),
            "goal_reached": bool(goal_reached),
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, False, truncated, info

    def _compute_reward(self, x_vel, action, x_pos, goal_reached):
        forward = self._forward_reward_weight * x_vel
        ctrl = self.control_cost(action)
        # distance error penalty (scaled)
        distance_error = np.abs(self._target_distance - x_pos)
        error_penalty = -0.05 * distance_error
        bonus = self._goal_bonus if goal_reached else 0.0
        reward = forward - ctrl + error_penalty + bonus
        reward_info = {
            "reward_forward": forward,
            "reward_ctrl": -ctrl,
            "reward_error": error_penalty,
            "reward_bonus": bonus,
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
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.standard_normal(
            self.model.nv
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def render(self, *args, **kwargs):
        return super().render(*args, **kwargs)

    def close(self):
        super().close()
