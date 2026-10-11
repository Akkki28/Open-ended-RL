import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class HalfCheetahEnergyEfficiencyEnv(MujocoEnv, utils.EzPickle):
    """HalfCheetah with an energy‑efficiency and posture reward.

    The observation and action spaces are identical to the standard
    HalfCheetah.  The reward is a weighted sum of forward speed, control
    effort, torso height, and pitch stability.  Episodes terminate after
    200 steps or when the pitch angle exceeds ±0.5 rad.
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"]}

    def __init__(
        self,
        xml_file: str = "half_cheetah.xml",
        frame_skip: int = 5,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.1,
        energy_weight: float = 0.5,
        height_weight: float = 0.5,
        pitch_weight: float = 0.5,
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
            energy_weight,
            height_weight,
            pitch_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            max_episode_steps,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._energy_weight = energy_weight
        self._height_weight = height_weight
        self._pitch_weight = pitch_weight
        self._reset_noise_scale = reset_noise_scale
        self._max_episode_steps = max_episode_steps
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._episode_step = 0

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        # Observation space definition (identical to the base env)
        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - exclude_current_positions_from_observation
        )
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)

    # ---------------------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------------------
    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    def energy_penalty(self, action):
        return self._energy_weight * np.sum(np.square(action))

    def height_reward(self):
        # qpos[1] corresponds to the torso height in the HalfCheetah model.
        # We encourage staying near the initial height (init_qpos[1]).
        height = self.data.qpos[1]
        target = self.init_qpos[1]
        return self._height_weight * (-(height - target) ** 2)

    def pitch_penalty(self):
        # qpos[2] is the torso pitch angle.
        pitch = self.data.qpos[2]
        return -self._pitch_weight * (pitch ** 2)

    # ---------------------------------------------------------------------
    # Core Gymnasium API
    # ---------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        observation = self.reset_model()
        self._episode_step = 0
        info = self._get_reset_info()
        return observation, info

    def step(self, action):
        # Record x‑position before simulation for velocity computation
        x_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_after = self.data.qpos[0]
        x_velocity = (x_after - x_before) / self.dt

        observation = self._get_obs()
        reward = self._compute_reward(x_velocity, action)
        info = {
            "x_position": x_after,
            "x_velocity": x_velocity,
            "reward_forward": self._forward_reward_weight * x_velocity,
            "reward_ctrl": -self.control_cost(action),
            "reward_energy": -self.energy_penalty(action),
            "reward_height": self.height_reward(),
            "reward_pitch": self.pitch_penalty(),
        }

        # Termination condition: excessive pitch
        terminated = abs(self.data.qpos[2]) > 0.5
        # Truncation after max steps
        self._episode_step += 1
        truncated = (self._episode_step >= self._max_episode_steps) and not terminated

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def _compute_reward(self, x_velocity, action):
        forward = self._forward_reward_weight * x_velocity
        ctrl = -self.control_cost(action)
        energy = -self.energy_penalty(action)
        height = self.height_reward()
        pitch = self.pitch_penalty()
        return forward + ctrl + energy + height + pitch

    def render(self, *args, **kwargs):
        # MujocoEnv provides a default render implementation.
        return super().render(*args, **kwargs)

    def close(self):
        return super().close()

    # ---------------------------------------------------------------------
    # Inherited helpers (unchanged from base HalfCheetah)
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
