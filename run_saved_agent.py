from pathlib import Path

def discover_agents(runs_dir="runs", environment_id=None):
    """Return a list of (env_id, checkpoint_path) tuples from runs_dir."""
    runs_path = Path(runs_dir)
    if not runs_path.is_dir():
        return []

    checkpoints = {}
    for checkpoint in runs_path.rglob("*.cleanrl_model"):
        run_name = checkpoint.parent.name
        parts = run_name.split("__")
        if not parts:
            continue
        env = parts[0]
        if environment_id is not None and env != environment_id:
            continue
        previous = checkpoints.get(env)
        if previous is None or checkpoint.stat().st_mtime > previous.stat().st_mtime:
            checkpoints[env] = checkpoint

    return [(env, path) for env, path in sorted(checkpoints.items())]

def run_agent(env_id, checkpoint_path, episodes=1, seed=1, cuda=False, algorithm="sac"):
    """Placeholder helper function for running/evaluating a saved agent."""
    pass
