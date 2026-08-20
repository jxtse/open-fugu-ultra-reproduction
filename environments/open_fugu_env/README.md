# open-fugu-env

A v1 verifiers environment, scaffolded with `init`.

## Develop

1. Implement `load` and the `@reward` in `open_fugu_env/taskset.py` (see `environments/`).
2. Install + run:

```bash
uv pip install -e .        # install this package (or register it in your project)
uv run eval open-fugu-env -n 3    # evaluate a few tasks with the bash harness
```

## Layout

- `open_fugu_env/taskset.py` — the task (`@reward` scoring + behavior) and the taskset: `load` (data + prompts).

Tune knobs from the CLI: `--env.taskset.num-tasks 10`, `--model <id>`, `-n`, and `-r`.
