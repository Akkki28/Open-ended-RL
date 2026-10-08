import os
import json
import ast
import hashlib
import importlib.util
import inspect
import re
import math

# Generated environments are written as importable Gymnasium modules.
output_dir = os.path.join("gymnasium", "envs", "classic_control")
os.makedirs(output_dir, exist_ok=True)

# System prompt from LLM.py
system_prompt = """
You are an expert in python programming and Reinforcement Learning. Your goal is to provide the next task for an agent looking to learn a collection of tasks in an open-ended fashion. You will be provided with a list of tasks and how well the agent does well there as compared to a random agent. Your task is to analyze the current level of the agent and write code for the next environment the agent should learn via RL. You are only allowed to change the reward functions and the initial configurations, not the naturer of the robot/agent(action/observation space).
The suggested task must be:
1) Interesting: This is the most important. The suggested task must seem interesting as stand alone tasks to a human user. Do not add random add ons but focus on building tasks that have semantic significance for learning.
2) Learnable: not too difficult for the agent based on its current level.
3) Novel: not already present in the existing environment list.
4) Similar: the observation and the action space must exactly the same as the initial task.

Return a complete Python module for one new environment.
The module must import numpy and gymnasium, define a class inheriting from
gymnasium.Env, keep the action_space and observation_space the same for all tasks, and implement
reset(seed=None, options=None), step(action), render(), and close() with changes making the tasks  more interesting/diffiult. Follow
the Gymnasium API: reset returns (observation, info), and step returns
(observation, reward, terminated, truncated, info). Use only dependencies
available in this repository plus numpy. Include a module-level metadata
dictionary and a clear class name ending in Env.
"""

# Example code from c.py to provide as reference

initial_environment = """
import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class HalfCheetahEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

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

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - exclude_current_positions_from_observation
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 1 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 1 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        info = {"x_position": x_position_after, "x_velocity": x_velocity, **reward_info}

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, False, False, info

    def _get_rew(self, x_velocity: float, action):
        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self.control_cost(action)

        reward = forward_reward - ctrl_cost

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }
        return reward, reward_info

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
        }


"""

GROQ_TPM_LIMIT = 8_000
GROQ_REQUEST_SAFETY_MARGIN = 200
MAX_COMPLETION_TOKENS = 4_500
MIN_COMPLETION_TOKENS = 3_000
MAX_GENERATION_ATTEMPTS = 3


def _estimate_token_count(*parts):
    """Conservatively estimate tokens without adding a tokenizer dependency."""
    return sum(len(part) for part in parts) // 3


def _completion_token_budget(system_content, user_content):
    """Keep Groq's prompt-plus-completion request below the organization limit."""
    prompt_tokens = _estimate_token_count(system_content, user_content)
    available_tokens = (
        GROQ_TPM_LIMIT - GROQ_REQUEST_SAFETY_MARGIN - prompt_tokens
    )
    if available_tokens < MIN_COMPLETION_TOKENS:
        raise RuntimeError(
            "The environment-generation prompt is too large for Groq's "
            f"{GROQ_TPM_LIMIT}-token TPM limit "
            f"(estimated prompt size: {prompt_tokens} tokens)"
        )
    return min(MAX_COMPLETION_TOKENS, available_tokens)


def _check_generated_environment(module_path):
    """Validate a generated module with Gymnasium's environment checker."""
    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env

    module_name = f"_generated_environment_{hashlib.sha1(module_path.encode()).hexdigest()}"
    module_spec = importlib.util.spec_from_file_location(module_name, module_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Unable to load generated environment module: {module_path}")

    module = importlib.util.module_from_spec(module_spec)
    try:
        module_spec.loader.exec_module(module)
        environment_classes = [
            item
            for _, item in inspect.getmembers(module, inspect.isclass)
            if issubclass(item, gym.Env)
            and item is not gym.Env
            and item.__module__ == module.__name__
        ]
        if len(environment_classes) != 1:
            raise RuntimeError(
                f"Expected exactly one generated Env class in {module_path}, "
                f"found {len(environment_classes)}"
            )

        environment = environment_classes[0]()
        try:
            check_env(environment, warn=True, skip_render_check=True)
        finally:
            environment.close()
    except Exception as error:
        raise RuntimeError(
            f"Generated environment failed Gymnasium's check_env: {module_path}"
        ) from error


# Track previously generated and learned environment concepts.
def generate_environment(learned_titles, performance_report, output_dir=None):
    """Generate and save one environment from measured agent performance."""
    from groq import BadRequestError, Groq

    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY must be set before generating an environment")

    output_dir = output_dir or os.path.join("gymnasium", "envs", "classic_control")
    os.makedirs(output_dir, exist_ok=True)
    user_prompt = f"""
Here is the base environment from which learning started
reference:
```python
{initial_environment}
```

The agents have currently successfully learned the following environments:
{learned_titles if learned_titles else "None so far."}

Measured random-agent versus learned-agent performance history:
{json.dumps(performance_report, indent=2)}

The next environment must keep the exact same observation and action spaces,
because it will be trained by the same DDPG implementation.

Please reason in no more than 80 words about what RL environment the agents
should learn next. Keep the generated module concise (preferably under 180
lines) so the complete response fits in the output limit.
Output one valid JSON object with string fields 'reasoning', 'task', and 'code'.
The code field must contain one complete Python source module, with a gymnasium.Env
class implementing reset, step, render, and close. Do not truncate the module, omit methods, or wrap the JSON in Markdown fences.
The environment must enforce a
200-step episode horizon: reset must set an episode step counter to zero, step must
increment it, and return truncated=True when the counter reaches 200 unless the
episode has already terminated. Keep the truncation logic inside the environment
as a defensive fallback; the runner also applies Gymnasium's TimeLimit wrapper.
"""
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    validation_feedback = ""
    last_error = None

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        attempt_prompt = user_prompt
        if validation_feedback:
            attempt_prompt += f"""

The previous generated environment failed validation. Fix this error in the
replacement module and return the complete JSON response again:
{validation_feedback}
"""
        completion_token_budget = _completion_token_budget(
            system_prompt, attempt_prompt
        )

        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                max_completion_tokens=completion_token_budget,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": attempt_prompt},
                ],
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content or "{}")
            title = result.get("task", "generated_environment")
            code_to_run = result.get("code", "")
            code_to_run = re.sub(
                r"^```(?:python)?\s*|\s*```$", "", code_to_run.strip()
            )
            if not code_to_run:
                raise RuntimeError("Groq returned no environment code")

            module_tree = ast.parse(code_to_run)
            environment_classes = [
                node
                for node in module_tree.body
                if isinstance(node, ast.ClassDef)
                and any(
                    isinstance(base, ast.Attribute) and base.attr == "Env"
                    for base in node.bases
                )
            ]
            required_methods = {"reset", "step", "render", "close"}
            if not environment_classes:
                raise RuntimeError("Groq returned code without a gymnasium.Env class")
            method_names = {
                node.name
                for node in environment_classes[0].body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            missing_methods = required_methods - method_names
            if missing_methods:
                raise RuntimeError(
                    "Groq returned an incomplete environment; missing methods: "
                    + ", ".join(sorted(missing_methods))
                )

            module_name = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
            module_name = module_name or "generated_environment"
            if len(module_name) > 80:
                title_hash = hashlib.sha1(title.encode("utf-8")).hexdigest()[:10]
                module_name = f"{module_name[:69].rstrip('_')}_{title_hash}"
            module_path = os.path.join(output_dir, f"{module_name}.py")
            with open(module_path, "w", encoding="utf-8") as environment_file:
                environment_file.write(code_to_run.rstrip() + "\n")

            _check_generated_environment(module_path)
        except Exception as error:
            if isinstance(error, BadRequestError) and "json_validate_failed" not in str(
                error
            ):
                raise
            last_error = error
            validation_feedback = str(error)
            if attempt < MAX_GENERATION_ATTEMPTS:
                continue
            raise RuntimeError(
                "Generated environment failed validation after "
                f"{MAX_GENERATION_ATTEMPTS} attempts: {error}"
            ) from error

        return {
            "title": title,
            "reasoning": result.get("reasoning", ""),
            "module_name": module_name,
            "module_path": module_path,
        }

    raise RuntimeError("Environment generation failed") from last_error
