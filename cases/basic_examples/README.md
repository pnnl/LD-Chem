# Basic LD-Chem examples

This directory contains small LD-Chem cases for learning the model interface.

These cases are intentionally lightweight. They are not intended to reproduce full observational or paper workflows. Instead, they show how to:

- define an aerosol population,
- define or load environmental trajectory information,
- run LD-Chem,
- write model output,
- optionally make a simple diagnostic plot.

## Cases

| Case | Description |
| --- | --- |
| `parcel/` | Adiabatic parcel simulation |
| `les_trajectory/` | Trajectory-driven simulation with a synthetic LES-style trajectory |
| `pichamber/` | Chamber-style trajectory simulation with a synthetic trajectory |

## Running a case

From a case directory:

```bash
python run_case.py
```

To choose the output directory:

```bash
python run_case.py --output-dir outputs
```

To show diagnostic plots:

```bash
python run_case.py --plot
```

Generated outputs are written to `outputs/` by default and should not be committed to the repository.
