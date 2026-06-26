"""Generate LD-Chem inputs from sample HISCALE and FLEXPART files.

This helper preserves the scientific logic from the original HISCALE input
generation script while writing ensemble-compatible LD-Chem input files.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any


DEFAULT_SPLAT_SPECIES = {
    "BC": ["soot"],
    "OIN": ["Dust"],
    "SO4": ["sulfate_nitrate_org"],
    "NO3": ["nitrate_amine_org"],
    "OC": ["org28", "org30_43", "BB_SOA", "org_amines", "BB", "pyridine"],
    "IEPOX_SOA": ["IEPOX_SOA"],
}

# mass thresholds[class][0] is min mass fraction, mean initial mass fraction,
# and std of initial mass fraction. mass thresholds[class][1] are the species
# included in that class.
DEFAULT_MASS_THRESHOLDS = {
    "IEPOX_SOA": [[0.3, 0.5, 0.1], ["IEPOX_OS", "tetrol", "tetrol_olig", "IEPOX_OH_SOA"]],
    "SO4": [[0.5, 0.7, 0.1], ["SO4"]],
    "NO3": [[0.5, 0.7, 0.1], ["NO3"]],
    "OC": [[0.5, 0.7, 0.1], ["OC"]],
    "BC": [[0.5, 0.7, 0.1], ["BC"]],
    "OIN": [[0.5, 0.7, 0.1], ["OIN"]],
}

DEFAULT_GAS_NAMES = (
    "SO2",
    "O3",
    "H2O2",
    "IEPOX",
    "OH",
    "HNO3",
    "NO2",
    "NO",
    "NH3",
)


def _np():
    import numpy as np

    return np


def _required_input_paths(raw_data_dir: Path) -> list[Path]:
    return [
        raw_data_dir / "BEASD_G1_20160425155810_R2_HISCALE_001s.txt",
        raw_data_dir / "AIMMS20_G1_20160425155810_R2_HISCALE020h.txt",
        raw_data_dir / "Splat_Composition_25-Apr-2016.txt",
        raw_data_dir / "HiScaleAMS_G1_20160425_R0.txt",
        raw_data_dir / "FLEXPART_output_traj_0001.txt",
        raw_data_dir / "CIMS_data",
        raw_data_dir / "CIMS_data" / "AGFL_atmosphere.txt",
    ]


def _check_required_inputs(raw_data_dir: Path) -> None:
    missing = [path for path in _required_input_paths(raw_data_dir) if not path.exists()]
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing required HISCALE sample preprocessing files:\n"
            f"{missing_list}\n\n"
            "Expected sample inputs under sample_inputs/HISCALE_data_0425/."
        )


def _write_pickle(path: Path, obj: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def build_aerosol_inputs(
    *,
    size_distribution_file: Path,
    aimms_file: Path,
    splat_file: Path,
    ams_file: Path,
    n_particles: int,
    z: float,
    dz: float,
    splat_cutoff_nm: float,
    random_seed: int,
    mean_pH: float,
    std_pH: float,
    size_dist_type: str,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Build aerosol population arrays from HISCALE observations."""
    np = _np()
    from part2pop.population import build_population

    population_settings = {
        "type": "hiscale_observations",
        "N_particles": n_particles,
        "beasd_file": str(size_distribution_file),
        "aimms_file": str(aimms_file),
        "splat_file": str(splat_file),
        "ams_file": str(ams_file),
        "z": z,
        "dz": dz,
        "splat_cutoff_nm": splat_cutoff_nm,
        "splat_species": DEFAULT_SPLAT_SPECIES,
        "mass_thresholds": DEFAULT_MASS_THRESHOLDS,
    }

    aerosol_population = build_population(population_settings)
    aero_spec_names = np.array([species.name for species in aerosol_population.species])
    aero_spec_masses = np.array(aerosol_population.spec_masses)
    num_concs = np.array(aerosol_population.num_concs)
    rng = np.random.default_rng(random_seed)
    pHs = rng.normal(size=num_concs.shape[0], loc=mean_pH, scale=std_pH)
    diameters = aerosol_population.get_particle_var("dry_diameter")

    total_masses = np.repeat(
        np.sum(aero_spec_masses, axis=1)[:, np.newaxis],
        aero_spec_masses.shape[1],
        axis=1,
    )
    aero_spec_fracs = aero_spec_masses / total_masses
    assert np.isclose(np.sum(aero_spec_fracs, axis=1), 1.0).all()

    evaluate_size_distribution(
        size_distribution_file=size_distribution_file,
        aimms_file=aimms_file,
        z=z,
        dz=dz,
        sampled_diameters=diameters,
        sampled_num_concs=num_concs,
    )
    evaluate_mass_concentrations(
        ams_file=ams_file,
        aimms_file=aimms_file,
        size_dist_type=size_dist_type,
        size_distribution_file=size_distribution_file,
        z=z,
        dz=dz,
        aero_spec_names=aero_spec_names,
        aero_spec_masses=aero_spec_masses,
        num_concs=num_concs,
    )

    return aero_spec_names, aero_spec_masses, aero_spec_fracs, num_concs, diameters, pHs


def evaluate_size_distribution(
    *,
    size_distribution_file: Path,
    aimms_file: Path,
    z: float,
    dz: float,
    sampled_diameters: Any,
    sampled_num_concs: Any,
) -> None:
    """Print diagnostics comparing measured and sampled size distributions."""
    np = _np()
    import part2pop.population.factory.helpers.hiscale as hiscale_helpers

    dp_lo_nm, dp_hi_nm, n_cm3, _ = hiscale_helpers._read_beasd_avg_size_dist(
        beasd_file=str(size_distribution_file),
        aimms_file=str(aimms_file),
        z=z,
        dz=dz,
    )
    dp_mid_nm = dp_lo_nm + 0.5 * (dp_hi_nm - dp_lo_nm)
    dp_mid_m = dp_mid_nm * 1e-9
    n_m3 = n_cm3 * 1e6
    dln = np.log(dp_hi_nm / dp_lo_nm)
    if np.any(~np.isfinite(dln)) or np.any(dln <= 0):
        raise ValueError("Invalid FIMS bin edges; cannot compute dln widths.")

    measured_surface_area = np.sum(4.0 * np.pi * np.power(dp_mid_m / 2, 2) * n_m3)
    measured_volume = np.sum((4.0 / 3.0) * np.pi * np.power(dp_mid_m / 2, 3) * n_m3)
    measured_mean_size = np.average(1e-9 * dp_mid_nm, weights=n_m3)

    sampled_surface_area = np.sum(
        4.0 * np.pi * np.power(sampled_diameters / 2, 2) * sampled_num_concs
    )
    sampled_volume = np.sum(
        (4.0 / 3.0) * np.pi * np.power(sampled_diameters / 2, 3) * sampled_num_concs
    )
    sampled_mean_size = np.average(sampled_diameters, weights=sampled_num_concs)

    print("SIZE DISTRIBUTION DIAGNOSTICS: (measured, sampled)")
    print(
        f"Total surface area (m^2/m^3): "
        f"{measured_surface_area:.2e}, {sampled_surface_area:.2e}"
    )
    print(f"Total volume (m^3/m^3): {measured_volume:.2e}, {sampled_volume:.2e}")
    print(f"Mean size (m): {measured_mean_size:.2e}, {sampled_mean_size:.2e}\n")


def evaluate_mass_concentrations(
    *,
    ams_file: Path,
    aimms_file: Path,
    size_dist_type: str,
    size_distribution_file: Path,
    z: float,
    dz: float,
    aero_spec_names: Any,
    aero_spec_masses: Any,
    num_concs: Any,
) -> None:
    """Print diagnostics comparing measured and sampled mass fractions."""
    np = _np()
    import part2pop.population.factory.helpers.hiscale as hiscale_helpers

    measured_mass_frac, _, _, _ = hiscale_helpers._read_ams_mass_fractions(
        ams_file=str(ams_file),
        aimms_file=str(aimms_file),
        size_dist_type=size_dist_type,
        size_dist_file=str(size_distribution_file),
        z=z,
        dz=dz,
    )

    sampled_masses = {"total": 0.0}
    for species_name in measured_mass_frac:
        if species_name == "OC":
            indices = [
                list(aero_spec_names).index(name)
                for name in DEFAULT_MASS_THRESHOLDS["IEPOX_SOA"][1]
            ]
            soa_masses = np.sum(np.sum(aero_spec_masses[:, indices], axis=1) * num_concs)
            oc_idx = np.where(aero_spec_names == species_name)[0][0]
            oc_masses = np.sum(aero_spec_masses[:, oc_idx] * num_concs)
            sampled_masses["OC"] = soa_masses + oc_masses
            sampled_masses["total"] += soa_masses + oc_masses
        else:
            idx = np.where(aero_spec_names == species_name)[0][0]
            sampled_masses[species_name] = np.sum(aero_spec_masses[:, idx] * num_concs)
            sampled_masses["total"] += sampled_masses[species_name]

    print("BULK MASS FRACTION DIAGNOSTICS: (measured, sampled)")
    for species_name in measured_mass_frac:
        sampled_frac = sampled_masses[species_name] / sampled_masses["total"]
        print(f"{species_name}: {measured_mass_frac[species_name]:.3f}, {sampled_frac:.3f}")
    print()


def read_flexpart_output(flexpart_file: Path) -> dict[str, Any]:
    """Read a FLEXPART text trajectory and return an LD-Chem trajectory dict."""
    np = _np()
    data = np.loadtxt(flexpart_file)

    trajectory = {
        "t": data[:, 1] * 3600.0,
        "x": data[:, 2],
        "y": data[:, 3],
        "z": data[:, 4],
        "P": data[:, 6] * 100.0,
        "T": data[:, 7],
    }

    qvapor = data[:, 10]
    temp_c = data[:, 7] - 273.15
    pressure_pa = data[:, 6] * 100.0
    saturation_vapor_pressure = 611.2 * np.exp(17.67 * temp_c / (temp_c + 243.5))
    saturation_mixing_ratio = (
        622.0
        * saturation_vapor_pressure
        / (pressure_pa - (1.0 - 0.622) * saturation_vapor_pressure)
    )
    relative_humidity = 100.0 * (qvapor / saturation_mixing_ratio)
    trajectory["s"] = relative_humidity / 100.0

    return trajectory


def _read_csv_table(path: Path) -> dict[str, Any]:
    np = _np()
    raw_data = np.loadtxt(path, delimiter=",", dtype="str")
    return {
        str(raw_data[0, column]): np.array(raw_data[1:, column], dtype="float64")
        for column in range(raw_data.shape[1])
    }


def _gas_name_from_cims_file(path: Path) -> str | None:
    parts = path.name.split("_")
    if len(parts) < 3 or not path.name.endswith("_obs.txt"):
        return None
    return parts[-2]


def _find_cims_file(gas_phase_directory: Path, gas: str) -> Path | None:
    for path in sorted(gas_phase_directory.iterdir()):
        if path.is_file() and _gas_name_from_cims_file(path) == gas:
            return path
    return None


def get_agfl_profile(gas: str, gas_phase_directory: Path) -> tuple[Any, Any]:
    """Read an AGFL fallback vertical gas profile."""
    np = _np()
    raw_data = np.loadtxt(gas_phase_directory / "AGFL_atmosphere.txt", dtype="str")
    agfl = {
        str(raw_data[0, column]): np.array(raw_data[1:, column], dtype="float64")
        for column in range(raw_data.shape[1])
    }
    return 1000.0 * agfl["z"], 1000.0 * agfl[gas]


def _median_profile(
    gas_data: dict[str, Any],
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> tuple[Any, Any]:
    np = _np()
    valid_idx = np.where(gas_data["Value_ppb"] >= 0.0)[0]
    alts = np.zeros(0)
    medians = np.zeros(0)

    if len(valid_idx) == 0:
        return alts, medians

    alt_grid = np.linspace(np.min(gas_data["Alt"][valid_idx]), np.max(gas_data["Alt"][valid_idx]), 11)
    alt_mids = 0.5 * (alt_grid[:-1] + alt_grid[1:])
    for upper_idx in range(1, len(alt_grid)):
        idx = np.where(
            (gas_data["Alt"] > alt_grid[upper_idx - 1])
            & (gas_data["Alt"] <= alt_grid[upper_idx])
            & (gas_data["Value_ppb"] >= 0.0)
            & (gas_data["Long"] > lon_min)
            & (gas_data["Long"] < lon_max)
            & (gas_data["Lat"] > lat_min)
            & (gas_data["Lat"] < lat_max)
        )[0]
        if len(idx) > 0:
            alts = np.append(alts, alt_mids[upper_idx - 1])
            medians = np.append(medians, np.percentile(gas_data["Value_ppb"][idx], 50))

    return alts, medians


def _no2_from_nox_no(gas_phase_directory: Path) -> dict[str, Any] | None:
    nox_file = _find_cims_file(gas_phase_directory, "NOx")
    no_file = _find_cims_file(gas_phase_directory, "NO")
    if nox_file is None or no_file is None:
        return None

    nox_data = _read_csv_table(nox_file)
    no_data = _read_csv_table(no_file)
    return {
        "Alt": nox_data["Alt"],
        "Lat": nox_data["Lat"],
        "Long": nox_data["Long"],
        "Value_ppb": nox_data["Value_ppb"] - no_data["Value_ppb"],
    }


def vertical_gas_profiles(
    *,
    gas_phase_directory: Path,
    gas_names: tuple[str, ...],
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> dict[str, dict[str, Any]]:
    """Build median vertical gas profiles from CIMS data with AGFL fallback."""
    gas_data_all = {}

    for gas in gas_names:
        if gas == "NO2":
            gas_data = _no2_from_nox_no(gas_phase_directory)
            if gas_data is None:
                gas_data = {}
        else:
            cims_file = _find_cims_file(gas_phase_directory, gas)
            gas_data = _read_csv_table(cims_file) if cims_file is not None else {}

        if gas_data:
            alts, medians = _median_profile(
                gas_data,
                lon_min=lon_min,
                lon_max=lon_max,
                lat_min=lat_min,
                lat_max=lat_max,
            )
            if len(medians) > 0:
                gas_data_all[gas] = {"ppb": medians, "alt": alts}
                continue

        try:
            alts, medians = get_agfl_profile(gas, gas_phase_directory)
        except Exception as exc:
            raise ValueError(f"No gas phase data for {gas}") from exc

        gas_data_all[gas] = {"ppb": medians, "alt": alts}

    return gas_data_all


def build_gas_trajectory(trajectory_z: Any, vertical_gas_data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Interpolate vertical gas profiles along a FLEXPART trajectory."""
    np = _np()
    from scipy.optimize import curve_fit

    gas_trajectory = {}
    for gas, profile in vertical_gas_data.items():
        ppb_interp = np.zeros(len(trajectory_z))
        for idx, altitude in enumerate(trajectory_z):
            if altitude < np.min(profile["alt"]):
                func = lambda x, a, b: a * x**b
                params, _ = curve_fit(func, profile["alt"][:2], profile["ppb"][:2], p0=[1, 0.1])
                ppb_interp[idx] = func(altitude, params[0], params[1])
            else:
                ppb_interp[idx] = np.interp(altitude, xp=profile["alt"], fp=profile["ppb"])
        gas_trajectory[gas] = ppb_interp

    return gas_trajectory


def preprocess_inputs(
    raw_data_dir: Path = Path("sample_inputs/HISCALE_data_0425"),
    output_dir: Path = Path("model_inputs/generated_inputs"),
    n_particles: int = 100,
    z: float = 100.0,
    dz: float = 100.0,
    splat_cutoff_nm: float = 85.0,
    mean_pH: float = 2.28,
    std_pH: float = 0.78,
    random_seed: int = 0,
    size_dist_type: str = "BEASD",
    lon_min: float = -97.5,
    lon_max: float = -97.4,
    lat_min: float = 36.05,
    lat_max: float = 36.81,
    gas_names: tuple[str, ...] = DEFAULT_GAS_NAMES,
) -> None:
    """Generate ensemble-compatible LD-Chem inputs from sample HISCALE files."""
    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)
    _check_required_inputs(raw_data_dir)

    size_distribution_file = raw_data_dir / "BEASD_G1_20160425155810_R2_HISCALE_001s.txt"
    aimms_file = raw_data_dir / "AIMMS20_G1_20160425155810_R2_HISCALE020h.txt"
    splat_file = raw_data_dir / "Splat_Composition_25-Apr-2016.txt"
    ams_file = raw_data_dir / "HiScaleAMS_G1_20160425_R0.txt"
    flexpart_file = raw_data_dir / "FLEXPART_output_traj_0001.txt"
    gas_phase_directory = raw_data_dir / "CIMS_data"

    (
        aero_spec_names,
        aero_spec_masses,
        aero_spec_fracs,
        num_concs,
        diameters,
        pHs,
    ) = build_aerosol_inputs(
        size_distribution_file=size_distribution_file,
        aimms_file=aimms_file,
        splat_file=splat_file,
        ams_file=ams_file,
        n_particles=n_particles,
        z=z,
        dz=dz,
        splat_cutoff_nm=splat_cutoff_nm,
        random_seed=random_seed,
        mean_pH=mean_pH,
        std_pH=std_pH,
        size_dist_type=size_dist_type,
    )

    flexpart_trajectory = read_flexpart_output(flexpart_file)
    vertical_gas_data = vertical_gas_profiles(
        gas_phase_directory=gas_phase_directory,
        gas_names=gas_names,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
    )
    flexpart_trajectory["gas"] = build_gas_trajectory(flexpart_trajectory["z"], vertical_gas_data)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_pickle(output_dir / "aero_spec_names.pkl", [aero_spec_names])
    _write_pickle(output_dir / "aero_spec_masses.pkl", [aero_spec_masses])
    _write_pickle(output_dir / "aero_spec_fracs.pkl", [aero_spec_fracs])
    _write_pickle(output_dir / "number_concentrations.pkl", [num_concs])
    _write_pickle(output_dir / "diameters.pkl", [diameters])
    _write_pickle(output_dir / "pHs.pkl", [pHs])
    _write_pickle(output_dir / "FLEXPART_trajectories.pkl", [flexpart_trajectory])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LD-Chem inputs from sample HISCALE/FLEXPART files."
    )
    parser.add_argument("--raw-data-dir", type=Path, default=Path("sample_inputs/HISCALE_data_0425"))
    parser.add_argument("--output-dir", type=Path, default=Path("model_inputs/generated_inputs"))
    parser.add_argument("--n-particles", type=int, default=100)
    parser.add_argument("--z", type=float, default=100.0)
    parser.add_argument("--dz", type=float, default=100.0)
    parser.add_argument("--mean-pH", type=float, default=2.28)
    parser.add_argument("--std-pH", type=float, default=0.78)
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_inputs(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        n_particles=args.n_particles,
        z=args.z,
        dz=args.dz,
        mean_pH=args.mean_pH,
        std_pH=args.std_pH,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
