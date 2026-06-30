# Basic LES trajectory case

This case demonstrates a trajectory-driven LD-Chem simulation.

The case defines a simple aerosol population and a synthetic LES-style trajectory with time-varying altitude, temperature, pressure, and saturation ratio. It then runs `simulate_les_trajectory`.

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

This case generates a synthetic trajectory inside `run_case.py`.

No external observational, WRF, FLEXPART, or ARM data are required.

The trajectory dictionary follows the same basic schema expected by LD-Chem trajectory simulations:

```text
t    time [s]
x    longitude [degrees east]
y    latitude [degrees north]
z    altitude [m]
T    temperature [K]
P    pressure [Pa]
s    saturation ratio [-]
gas  gas-phase concentration time series, when available
```

Users can replace the synthetic trajectory with preprocessed WRF/FLEXPART trajectory data if they have data in the expected schema.

## Outputs

By default, this case writes:

```text
outputs/trajectory.pkl
outputs/trajectory_restart.pkl
```

Generated outputs should not be committed to the repository.
