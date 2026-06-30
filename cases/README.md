# LD-Chem cases

This directory contains runnable LD-Chem cases.

A case is a complete example configuration that shows how to set up and run LD-Chem for a specific purpose. Some cases are small instructional examples. Others document larger research workflows.

## Basic examples

`basic_examples/` contains small cases intended for new users. These cases use simple prescribed or synthetic inputs and are designed to demonstrate the LD-Chem interface without requiring observational, WRF, FLEXPART, or ARM data.

Available basic examples:

- `basic_examples/parcel/`: adiabatic parcel simulation
- `basic_examples/les_trajectory/`: trajectory-driven simulation using a simple synthetic trajectory
- `basic_examples/pichamber/`: chamber-style trajectory simulation using a simple synthetic trajectory

Each basic example can be run from its directory with:

```bash
python run_case.py
```

Most basic examples also support:

```bash
python run_case.py --plot
python run_case.py --output-dir outputs
```

## Research cases

Research cases document larger workflows connected to specific scientific applications.

Available research cases:

* `hiscale_20160425/`: HISCALE April 25, 2016 trajectory ensemble workflow associated with Beeler et al. 2026

Large observational, WRF, FLEXPART, and generated output files are not stored directly in this GitHub repository unless they are small enough for practical use. Research cases may point to external data archives and may include preprocessed LD-Chem inputs for convenience.
