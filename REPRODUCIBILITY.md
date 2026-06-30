# Reproducibility

## Install

From a clean clone:

```bash
git clone https://github.com/pnnl/LD-Chem.git
cd LD-Chem
python -m pip install -e ".[test]"
```

Optional conda bootstrap:

```bash
conda env create -f environment.yml
conda activate ld-chem
```

The editable pip install is the preferred development and testing path because
it uses the package metadata in `pyproject.toml`.

## Tests

Run the default test suite with:

```bash
python -m pytest
```

Integration tests can be run explicitly with:

```bash
python -m pytest tests/integration
```

## Cases

The README quick start is a lightweight smoke test using a small synthetic
aerosol population. Runnable cases under `cases/` demonstrate larger usage
patterns and may write runtime output files in the current working directory.

## Data Availability

This repository bundles source code, tests, cases, default mechanisms, and
small runtime species/mechanism data files. It does not bundle an unpublished
paper-specific dataset.

For publication, paper-specific datasets and large generated outputs should be
archived separately in an appropriate data repository. Final DOI or accession
values should be recorded in the manuscript Data Availability statement and in
release documentation only after they are minted.

## Manual Release Items

- Mint and record the final software release DOI after the release is created.
- Mint and record any paper dataset DOI or accession if required by the
  manuscript.
