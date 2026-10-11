import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class HalfCheetahSmoothControlEnv(MujocoEnv, utils.EzPickle):
    """HalfCheetah with smooth‑control and torso‑upright incentives.

    Reward components:
        * forward speed (weighted by ``forward_reward_weight``)
        * control effort (``ctrl_cost_weight``)
        * jerk (change in action, ``jerk_cost_weight``)
        * torso pitch deviation (``pitch_penalty_weight``)
    Episode length: 200 steps (handled internally via ``truncated`` flag).
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"]}

    def __init__(
        self,
        xml_file: str = "half_cheetah.xml",
        frame_skip: int = 5,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.1,
        jerk_cost_weight: float = 0.05,
        pitch_penalty_weight: float = 0.5,
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
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
            jerk_cost_weight,
            pitch_penalty_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            max_episode_steps,
            **kwargs,
        )
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._jerk_cost_weight = jerk_cost_weight
        self._pitch_penalty_weight = pitch_penalty_weight
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._max_episode_steps = max_episode_steps
        self._step_counter = 0
        self._prev_action = None
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )
        # Observation space definition matches the base HalfCheetah env
        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - exclude_current_positions_from_observation
        )
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)

    # ---------------------------------------------------------------------
    # Helper cost functions
    # ---------------------------------------------------------------------
    def control_cost(self, action: np.ndarray) -> float:
        return self._ctrl_cost_weight * np.sum(np.square(action))

    def jerk_cost(self, action: np.ndarray) -> float:
        if self._prev_action is None:
            return 0.0
        jerk = action - self._prev_action
        return self._jerk_cost_weight * np.sum(np.square(jerk))

    def pitch_penalty(self) -> float:
        # In the HalfCheetah model qpos[2] corresponds to torso pitch
        pitch = self.data.qpos[2]
        return self._pitch_penalty_weight * (pitch ** 2)

    # ---------------------------------------------------------------------
    # Gym API
    # ---------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._step_counter = 0
        self._prev_action = np.zeros(self.model.nu, dtype=np.float64)
        observation = self.reset_model()
        info = self._get_reset_info()
        return observation, info

    def step(self, action: np.ndarray):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        # Compute reward components
        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self.control_cost(action)
        jerk_cost = self.jerk_cost(action)
        pitch_pen = self.pitch_penalty()
        reward = forward_reward - ctrl_cost - jerk_cost - pitch_pen

        # Update previous action for next step's jerk calculation
        self._prev_action = action.copy()

        observation = self._get_obs()
        self._step_counter += 1
        terminated = False
        truncated = self._step_counter >= self._max_episode_steps
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_jerk": -jerk_cost,
            "reward_pitch": -pitch_pen,
            "step": self._step_counter,
        }
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def render(self):
        # Delegate to MujocoEnv's render implementation
        return super().render()

    def close(self):
        return super().close()

    # ---------------------------------------------------------------------
    # Observation handling (same as base HalfCheetah)
    # ---------------------------------------------------------------------
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

    def _get_reset_info(self):
        return {"x_position": self.data.qpos[0]}
