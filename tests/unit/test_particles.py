import pytest
import numpy as np
from pathlib import Path
from multipart.particles import AerosolSpecies, retrieve_one_species


def test_aerosol_species_creation():
    """Test AerosolSpecies dataclass creation."""
    species = AerosolSpecies(
        name="SO4",
        density=1800.0,
        kappa=0.65,
        molar_mass=0.096,
        surface_tension=0.072
    )

    assert species.name == "SO4"
    assert species.density == 1800.0
    assert species.kappa == 0.65
    assert species.molar_mass == 0.096
    assert species.surface_tension == 0.072


def test_retrieve_one_species():
    """Test aerosol species data retrieval from file."""
    # Path to species_data directory from test file location
    # tests/objects/test_particles.py -> ../../src/multipart/species_data/
    species_data_path = Path(__file__).parent.parent.parent / "src" / "multipart" / "species_data"

    # Test with a known species from aero_data.dat
    species = retrieve_one_species("SO4", specdata_path=str(species_data_path) + "/")

    assert species.name == "SO4"
    assert isinstance(species.density, float)
    assert isinstance(species.kappa, float)
    assert isinstance(species.molar_mass, float)
    assert species.surface_tension == 0.072  # default value

    # Test with another species
    species_oc = retrieve_one_species("OC", specdata_path=str(species_data_path) + "/")
    assert species_oc.name == "OC"
    assert species_oc.density == 1200.0  # from aero_data.dat
    assert species_oc.kappa == 0.1


def test_retrieve_one_species_custom_surface_tension():
    """Test aerosol species retrieval with custom surface tension."""
    species_data_path = Path(__file__).parent.parent.parent / "src" / "multipart" / "species_data"
    custom_surface_tension = 0.08
    species = retrieve_one_species(
        "SO4",
        specdata_path=str(species_data_path) + "/",
        surface_tension=custom_surface_tension
    )
    assert species.name == "SO4"
    assert species.surface_tension == custom_surface_tension
