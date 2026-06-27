# Model outputs

This directory is the default location for generated HISCALE LD-Chem model outputs.

Generated outputs should not be committed to the repository.

Typical generated files may include:

```text
trajectory_000000.pkl
trajectory_restart_000000.pkl
trajectory_000001.pkl
trajectory_restart_000001.pkl
...
```

Use `run_case.py --output-dir <path>` to write outputs somewhere else.
