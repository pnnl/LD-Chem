# HISCALE April 25, 2016 case

This case documents the LD-Chem workflow for the HISCALE April 25, 2016 trajectory ensemble associated with Beeler et al. 2026.

The main workflow starts from committed, LD-Chem-ready inputs in `model_inputs/` and writes generated model outputs to `model_outputs/`.

## Quick start

From this directory:

```bash
python run_case.py
```

This runs the LD-Chem ensemble using the committed model inputs in:

```text
model_inputs/
```

and writes generated model output files to:

```text
model_outputs/
```

To run only one simulation as a smoke test:

```bash
python run_case.py --run-ensemble --max-simulations 1
```

Generated model outputs should not be committed to the repository.

## Inputs

The committed model input files are stored directly in:

```text
model_inputs/
```

Expected files:

```text
FLEXPART_trajectories.pkl
diameters.pkl
aero_spec_names.pkl
aero_spec_fracs.pkl
aero_spec_masses.pkl
number_concentrations.pkl
pHs.pkl
```

These files are LD-Chem-ready inputs. They are the recommended entry point for this case.

## Optional advanced preprocessing

An advanced preprocessing helper is included for users who want to regenerate LD-Chem-ready inputs from upstream observational and FLEXPART files.

The upstream observational and FLEXPART files are not bundled with the GitHub repository. Users must provide them locally.

Default local layout:

```text
data/
  obs/
    BEASD_G1_20160425155810_R2_HISCALE_001s.txt
    AIMMS20_G1_20160425155810_R2_HISCALE020h.txt
    Splat_Composition_25-Apr-2016.txt
    HiScaleAMS_G1_20160425_R0.txt
    CIMS_data/
      AGFL_atmosphere.txt
      g1_20160425a_*_obs.txt
  flexpart/
    FLEXPART_output_traj_0001.txt
```

To generate LD-Chem-ready inputs from local upstream files:

```bash
python run_case.py --preprocess-inputs
```

This writes generated inputs to:

```text
generated_model_inputs/
```

To generate inputs and immediately run LD-Chem on them:

```bash
python run_case.py --preprocess-inputs --run-ensemble --max-simulations 1
```

To use custom upstream paths:

```bash
python run_case.py --preprocess-inputs \
  --obs-data-dir /path/to/obs \
  --flexpart-file /path/to/FLEXPART_output_traj_0001.txt
```

## Directory structure

```text
cases/hiscale_20160425/
  README.md
  DATA.md
  run_case.py
  helpers/
    preprocess_inputs.py
    run_ensemble.py
    postprocess_figures.py
  model_inputs/
    README.md
    *.pkl
  model_outputs/
    README.md
```

Local-only directories used by the optional advanced workflow are ignored by Git:

```text
data/
generated_model_inputs/
```

## Reproducibility scope

Included in GitHub:

* LD-Chem case driver,
* ensemble runner,
* optional preprocessing helper,
* committed LD-Chem-ready model inputs,
* documentation.

Stored externally or provided by the user:

* upstream observational files,
* CIMS/trace-gas profile files,
* WRF output files,
* FLEXPART-WRF output files,
* large generated LD-Chem outputs.

The full upstream WRF/FLEXPART workflow is not reproduced from the GitHub repository alone.
