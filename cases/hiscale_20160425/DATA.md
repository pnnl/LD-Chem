# HISCALE April 25, 2016 data

This file documents the data used by the HISCALE April 25, 2016 LD-Chem case.

## Committed LD-Chem-ready inputs

The main workflow starts from committed model input files in:

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

Each row or list entry corresponds to one LD-Chem trajectory simulation.

These files are already preprocessed for LD-Chem and are the recommended entry point for the case.

## Generated model outputs

Generated LD-Chem model outputs are written to:

```text
model_outputs/
```

Generated outputs are ignored by Git and should not be committed.

## Optional upstream preprocessing data

The optional preprocessing workflow can regenerate LD-Chem-ready inputs from upstream observational and FLEXPART files.

Those upstream files are not bundled with the GitHub repository.

Default local layout for optional preprocessing:

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

The observational inputs are used to initialize aerosol and gas-phase conditions. These files may come from ARM, campaign-specific IOP products, or derived observational files.

Aerosol population construction uses `part2pop`.

Generated LD-Chem-ready inputs from this optional preprocessing workflow are written to:

```text
generated_model_inputs/
```

Generated model inputs are ignored by Git.

## External archive

Large WRF-FLEXPART and supporting data files are stored outside this GitHub repository.

Archive location:

```text
https://portal.nersc.gov/archive/projects/m1657/www/Beeler_etal_2026
```

This link is not yet active.

ARM or campaign observational data can be found on the ARM data archive, but may also require separate access through the ARM Data Center or campaign-specific data products.

```
https://adc.arm.gov/discovery/results/iopShortName::sgp2016hiscale
```

## WRF-FLEXPART provenance

Recovered WRF-FLEXPART files indicate that the April 25, 2016 trajectory calculation used WRF domain 2 output files at one-minute resolution from 2016-04-25 15:00:00 through 2016-04-25 18:00:00 UTC.

The WRF files follow this pattern:

```text
wrfout_d02_2016-04-25_HH_MM_00.nc
```

A representative WRF output file reported:

```text
WRF version: WRF V4.4
Domain: d02
Horizontal grid spacing: 100 m
Projection: Lambert Conformal
Center latitude: 36.60227
Center longitude: -97.48846
Simulation start: 2016-04-25 12:00:00 UTC
```

FLEXPART-WRF input metadata indicated a forward simulation from 2016-04-25 15:00:00 to 2016-04-25 18:00:00 UTC.

The recovered `AVAILABLE` file listed 181 WRF d02 files from 15:00 through 18:00 inclusive.

## Data not committed to GitHub

The following should not be committed to GitHub:

```text
data/
generated_model_inputs/
model_outputs/trajectory_*.pkl
model_outputs/trajectory_restart_*.pkl
wrfout_*
wrfinput_*
wrfbdy_*
wrfrst_*
partposit_*
*.nc
```

Large files should be stored in the external archive and referenced from this documentation.

The full upstream WRF/FLEXPART workflow is not reproducible from the GitHub repository alone.
