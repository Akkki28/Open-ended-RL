import os
import json
import ast
import hashlib
import re
import math

# Generated environments are written as importable Gymnasium modules.
output_dir = os.path.join("gymnasium", "envs", "classic_control")
os.makedirs(output_dir, exist_ok=True)

# System prompt from LLM.py
system_prompt = """
You are an expert in python programming and Reinforcement Learning. Your goal is to provide the next task for an agent looking to learn a collection of tasks in an open-ended fashion. You will be provided with a list of tasks and how well the agent does well there as compared to a random agent. Your task is to analyze the current level of the agent and write code for the next environment the agent should learn via RL. You are only allowed to change the reward functions and the initial configurations, not the naturer of the robot/agent(action/observation space).
The suggested task must be:
1) Interesting: This is the most important. It should not have random add-ons, but rather the suggested task shall be traditionally catch te interest of a user.
2) Learnable: not too difficult for the agent based on its current level.
3) Novel: not already present in the existing environment list.
4) Similar: the observation and the action space must exactly the same as the initial task.
5) Diverse: vary dynamics, rewards, observations, actions, or initial conditions.

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
import math

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled


class Continuous_MountainCarEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: str | None = None, goal_velocity=0):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = (
            0.45  # was 0.5 in gymnasium, 0.45 in Arnaud de Broissia's version
        )
        self.goal_velocity = goal_velocity
        self.power = 0.0015

        self.low_state = np.array(
            [self.min_position, -self.max_speed], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_speed], dtype=np.float32
        )

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True

        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )

    def step(self, action: np.ndarray):
        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], self.min_action), self.max_action)

        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        if velocity > self.max_speed:
            velocity = self.max_speed
        if velocity < -self.max_speed:
            velocity = -self.max_speed
        position += velocity
        if position > self.max_position:
            position = self.max_position
        if position < self.min_position:
            position = self.min_position
        if position == self.min_position and velocity < 0:
            velocity = 0

        # Convert a possible numpy bool to a Python bool.
        terminated = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )

        reward = 0
        if terminated:
            reward = 100.0
        reward -= math.pow(action[0], 2) * 0.1

        self.state = np.array([position, velocity], dtype=np.float32)

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self.state, reward, terminated, False, {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(options, -0.6, -0.4)
        self.state = np.array([self.np_random.uniform(low=low, high=high), 0])

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def _height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e

        if self.screen is None:
            pygame.display.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array":
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.max_position - self.min_position
        scale = self.screen_width / world_width
        carwidth = 40
        carheight = 20

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        pos = self.state[0]

        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = self._height(xs)
        xys = list(zip((xs - self.min_position) * scale, ys * scale, strict=True))

        pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))

        clearance = 10

        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            coords.append(
                (
                    c[0] + (pos - self.min_position) * scale,
                    c[1] + clearance + self._height(pos) * scale,
                )
            )

        gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))

        for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            wheel = (
                int(c[0] + (pos - self.min_position) * scale),
                int(c[1] + clearance + self._height(pos) * scale),
            )

            gfxdraw.aacircle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )
            gfxdraw.filled_circle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )

        flagx = int((self.goal_position - self.min_position) * scale)
        flagy1 = int(self._height(self.goal_position) * scale)
        flagy2 = flagy1 + 50
        gfxdraw.vline(self.surf, flagx, flagy1, flagy2, (0, 0, 0))

        gfxdraw.aapolygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        gfxdraw.filled_polygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


"""

GROQ_TPM_LIMIT = 8_000
GROQ_REQUEST_SAFETY_MARGIN = 200
MAX_COMPLETION_TOKENS = 4_500
MIN_COMPLETION_TOKENS = 3_000


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


# Track previously generated and learned environment concepts.
def generate_environment(learned_titles, performance_report, output_dir=None):
    """Generate and save one environment from measured agent performance."""
    from groq import Groq

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

Measured random-agent versus learned-agent performance:
{json.dumps(performance_report, indent=2)}

The next environment must keep the exact same observation and action spaces,
because it will be trained by the same DDPG implementation.

Please reason briefly about what RL environment the agents should learn next.
Output one valid JSON object with string fields 'reasoning', 'task', and 'code'.
The code field must contain one complete Python source module, with a gymnasium.Env
class implementing reset, step, render, and close. Do not truncate the module,
omit methods, or wrap the JSON in Markdown fences. The environment must enforce a
200-step episode horizon: reset must set an episode step counter to zero, step must
increment it, and return truncated=True when the counter reaches 200 unless the
episode has already terminated. Keep the truncation logic inside the environment
as a defensive fallback; the runner also applies Gymnasium's TimeLimit wrapper.
"""
    completion_token_budget = _completion_token_budget(system_prompt, user_prompt)

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_completion_tokens=completion_token_budget,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content or "{}")
    title = result.get("task", "generated_environment")
    code_to_run = result.get("code", "")
    code_to_run = re.sub(r"^```(?:python)?\s*|\s*```$", "", code_to_run.strip())
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

    return {
        "title": title,
        "reasoning": result.get("reasoning", ""),
        "module_name": module_name,
        "module_path": module_path,
    }
