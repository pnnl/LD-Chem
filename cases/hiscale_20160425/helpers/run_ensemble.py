"""Run the HISCALE April 25, 2016 LD-Chem ensemble."""

from __future__ import annotations

from pathlib import Path
import pickle
import sys


REQUIRED_INPUT_FILES = [
    "FLEXPART_trajectories.pkl",
    "diameters.pkl",
    "aero_spec_names.pkl",
    "aero_spec_fracs.pkl",
    "aero_spec_masses.pkl",
    "number_concentrations.pkl",
    "pHs.pkl",
]


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _check_required_inputs(input_dir: Path) -> None:
    missing = [name for name in REQUIRED_INPUT_FILES if not (input_dir / name).exists()]
    if missing:
        missing_list = "\n".join(f"  - {name}" for name in missing)
        raise FileNotFoundError(
            f"Missing required LD-Chem-ready input files in {input_dir}:\n"
            f"{missing_list}\n\n"
            "Use the committed paper inputs in model_inputs/, or generate local inputs with:\n"
            "  python run_case.py --preprocess-inputs\n\n"
            "Generated inputs are written to generated_model_inputs/ and are not committed."
        )


def run_ensemble(
    input_dir: Path,
    output_dir: Path,
    max_simulations: int | None = None,
) -> None:
    """Run LD-Chem trajectory simulations from preprocessed HISCALE inputs."""
    repo_src = Path(__file__).resolve().parents[3] / "src"
    if repo_src.exists() and str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))

    import numpy as np
    from ld_chem.run import simulate_les_trajectory
    from part2pop.population import build_population

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _check_required_inputs(input_dir)

    diameters = _load_pickle(input_dir / "diameters.pkl")
    aero_spec_names = _load_pickle(input_dir / "aero_spec_names.pkl")
    aero_spec_fracs = _load_pickle(input_dir / "aero_spec_fracs.pkl")
    number_concentrations = _load_pickle(input_dir / "number_concentrations.pkl")
    aero_spec_masses = _load_pickle(input_dir / "aero_spec_masses.pkl")
    pHs = _load_pickle(input_dir / "pHs.pkl")
    trajectories = _load_pickle(input_dir / "FLEXPART_trajectories.pkl")

    n_simulations = len(trajectories)
    if max_simulations is not None:
        n_simulations = min(n_simulations, max_simulations)

    for simulation_number in range(n_simulations):
        sim_diameters = diameters[simulation_number]
        sim_aero_spec_names = aero_spec_names[simulation_number]
        sim_aero_spec_fracs = aero_spec_fracs[simulation_number]
        sim_num_concs = number_concentrations[simulation_number]
        sim_aero_spec_masses = aero_spec_masses[simulation_number]
        sim_pHs = pHs[simulation_number]
        sim_trajectory = trajectories[simulation_number]

        pop_cfg = {
            "type": "monodisperse",
            "N": sim_num_concs,
            "D": sim_diameters,
            "aero_spec_names": sim_aero_spec_names,
            "aero_spec_fracs": sim_aero_spec_fracs,
        }

        aerosol_population = build_population(pop_cfg)
        diameters_from_masses = aerosol_population.get_particle_var("dry_diameter")
        if not np.isclose(diameters_from_masses, sim_diameters).all():
            raise ValueError(
                f"Particle masses and diameters are inconsistent for "
                f"simulation {simulation_number}."
            )

        output_filename = output_dir / f"trajectory_{simulation_number:06d}.pkl"
        restart_filename = output_dir / f"trajectory_restart_{simulation_number:06d}.pkl"

        simulate_les_trajectory(
            sim_aero_spec_names[0],
            sim_aero_spec_masses,
            sim_num_concs,
            sim_pHs,
            sim_trajectory,
            dt=5.0,
            restart_filename=str(restart_filename),
            radius_scale="log",
            output_filename=str(output_filename),
            aq_chemistry=["IEPOX", "sulfate", "nitrate", "ammonium"],
            write_every=30.0,
            condensation=True,
            gas_chemistry=True,
            print_to_screen=True,
            cocondensation=False,
            relaxation_time=24.475,
        )
