# HISCALE April 25, 2016 data

This file documents the data used by the HISCALE April 25, 2016 LD-Chem case.

## External data archive

Large WRF-FLEXPART and supporting data files are stored outside this GitHub repository.

Archive location:

```text
https://portal.nersc.gov/archive/projects/m1657/www/Beeler_etal_2026
```

This link is not yet active.

## Preprocessed paper inputs

The main LD-Chem workflow starts from preprocessed model inputs in:

```text
model_inputs/paper_input_files/
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

## Sample preprocessing inputs

A small set of sample files is included under:

```text
sample_inputs/HISCALE_data_0425/
```

These files are used by:

```bash
python run_case.py --preprocess-inputs
```

Expected sample files:

```text
BEASD_G1_20160425155810_R2_HISCALE_001s.txt
AIMMS20_G1_20160425155810_R2_HISCALE020h.txt
Splat_Composition_25-Apr-2016.txt
HiScaleAMS_G1_20160425_R0.txt
FLEXPART_output_traj_0001.txt
CIMS_data/
```

The sample preprocessing workflow uses these files to build a one-trajectory LD-Chem input set in:

```text
model_inputs/generated_inputs/
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

The following file types should not be committed to GitHub:

```text
wrfout_*
wrfinput_*
wrfbdy_*
wrfrst_*
partposit_*
large LD-Chem output files
```

Large files should be stored in the external archive and referenced from this documentation.
