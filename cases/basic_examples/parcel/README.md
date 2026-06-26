# Basic parcel case

This case demonstrates an adiabatic parcel simulation with LD-Chem.

The case defines a simple two-mode aerosol population using `part2pop`, runs an LD-Chem parcel simulation, and writes the output to `outputs/trajectory.pkl`.

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

The aerosol population contains two externally mixed modes:

* sulfate particles,
* organic particles.

The pH values are prescribed deterministically for reproducibility.

## Outputs

By default, this case writes:

```text
outputs/trajectory.pkl
outputs/trajectory_restart.pkl
```

Generated outputs should not be committed to the repository.
