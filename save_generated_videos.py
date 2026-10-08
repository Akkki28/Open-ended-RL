"""Record videos of saved agents in all registered generated environments.

Each environment is processed independently.  A missing checkpoint, unsupported
render mode, or video encoder error is reported and does not prevent the other
environments from being attempted.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from run_ddpg import register_saved_generated_environments


def _load_registry(runs_dir: Path) -> dict[str, dict[str, Any]]:
    registry_path = runs_dir / "generated_environments.json"
    try:
        with registry_path.open(encoding="utf-8") as registry_file:
            registry = json.load(registry_file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read generated environment registry: {registry_path}") from error

    if not isinstance(registry, dict):
        raise RuntimeError(f"Generated environment registry must be an object: {registry_path}")
    return registry


def _discover_checkpoints(
    runs_dir: Path, environment_ids: set[str]
) -> list[tuple[str, Path]]:
    """Return the newest checkpoint found for each registered environment."""
    checkpoints: dict[str, Path] = {}
    for checkpoint in runs_dir.rglob("*.cleanrl_model"):
        run_name = checkpoint.parent.name
        environment_id = run_name.split("__", 1)[0]
        if environment_id not in environment_ids:
            continue

        previous = checkpoints.get(environment_id)
        if previous is None or checkpoint.stat().st_mtime > previous.stat().st_mtime:
            checkpoints[environment_id] = checkpoint

    return sorted(checkpoints.items())


def _build_actor(
    env_id: str, checkpoint_path: Path, algorithm: str, device: torch.device
) -> tuple[torch.nn.Module, gym.Env]:
    actor_env = gym.vector.SyncVectorEnv([lambda: gym.make(env_id)])
    try:
        if algorithm == "ddpg":
            from run_ddpg import Actor

            actor = Actor(actor_env)
        elif algorithm == "sac":
            from run_sac import Actor

            actor = Actor(actor_env)
        elif algorithm == "ppo":
            from run_ppo import Agent

            if not isinstance(actor_env.single_action_space, gym.spaces.Discrete):
                raise ValueError("PPO evaluation requires a discrete action space")
            actor = Agent(actor_env)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        actor.load_state_dict(checkpoint[0])
        actor.to(device)
        actor.eval()
        return actor, actor_env
    except Exception:
        actor_env.close()
        raise


def _action(
    actor: torch.nn.Module,
    observation: Any,
    algorithm: str,
    device: torch.device,
) -> np.ndarray | int:
    observation_tensor = torch.as_tensor(
        observation, dtype=torch.float32, device=device
    ).unsqueeze(0)
    with torch.no_grad():
        if algorithm == "ppo":
            return int(actor.get_action_and_value(observation_tensor)[0].item())
        if algorithm == "sac":
            action = actor.get_action(observation_tensor)[2]
        else:
            action = actor(observation_tensor)
    return action.squeeze(0).cpu().numpy()


def record_environment(
    env_id: str,
    checkpoint_path: Path,
    video_folder: Path,
    seed: int,
    episodes: int,
    algorithm: str,
    device: torch.device,
) -> None:
    """Record one environment; callers are responsible for handling failures."""
    actor, actor_env = _build_actor(env_id, checkpoint_path, algorithm, device)
    env = None
    try:
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_folder),
            episode_trigger=lambda episode_id: episode_id < episodes,
            name_prefix=f"saved-agent-{env_id}",
        )
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            episode_return = 0.0
            terminated = truncated = False

            while not (terminated or truncated):
                action = _action(actor, observation, algorithm, device)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)

            print(
                f"env={env_id}, episode={episode + 1}/{episodes}, "
                f"return={episode_return:.3f}"
            )
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            actor_env.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Save videos for the saved agents in generated environments."
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--video-folder", type=Path, default=Path("videos"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--algorithm", choices=("ddpg", "sac", "ppo"), default="sac")
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    registry = _load_registry(args.runs_dir)
    register_saved_generated_environments()
    agents = _discover_checkpoints(args.runs_dir, set(registry))
    if not agents:
        print(f"No checkpoints found for registered environments in {args.runs_dir}", file=sys.stderr)
        return 1

    args.video_folder.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    succeeded = 0
    failed = 0

    for env_id, checkpoint_path in agents:
        print(f"Recording {env_id} from {checkpoint_path}")
        try:
            record_environment(
                env_id,
                checkpoint_path,
                args.video_folder,
                args.seed,
                args.episodes,
                args.algorithm,
                device,
            )
        except Exception:
            failed += 1
            print(f"FAILED {env_id}; continuing with the remaining environments", file=sys.stderr)
            traceback.print_exc()
        else:
            succeeded += 1

    print(f"Video generation finished: {succeeded} succeeded, {failed} failed.")
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
