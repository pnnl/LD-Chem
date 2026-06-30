# Basic Pi Chamber case

This case demonstrates a chamber-style trajectory simulation with LD-Chem.

The case defines a simple aerosol population and a synthetic chamber trajectory with prescribed time-varying saturation ratio. It then runs the LD-Chem trajectory driver.

## Run the case

From this directory:

```bash
python run_case.py
```

To write outputs to a different directory:

```bash
python run_case.py --output-dir outputs
```

To show diagnostic plots after the run:

```bash
python run_case.py --plot
```

## Inputs

This case generates its inputs inside `run_case.py`.

No external observational, WRF, FLEXPART, or ARM data are required.

The chamber trajectory is intentionally simple and is intended only to demonstrate the LD-Chem interface.

## Outputs

By default, this case writes:

```text
outputs/trajectory.pkl
outputs/trajectory_restart.pkl
```

Generated outputs should not be committed to the repository.
