import pytest
import numpy as np
import warnings, tempfile, os
from pathlib import Path
from ld_chem.run import simulate_parcel, simulate_les_trajectory, restart_trajectory

pytestmark = pytest.mark.integration


def test_simulate_parcel_basic_parameters():
    """Test simulate_parcel with basic parameters."""
    # Set up minimal test data
    aero_spec_names = np.array([["SO4", "H2O"]])
    aero_spec_masses = np.array([[1e-15, 0.0]])  # Very small masses
    num_concs = np.array([1e6])
    pHs = np.array([5.0])

    # Use temporary directory for output files
    with tempfile.TemporaryDirectory() as temp_dir:
        output_filename = os.path.join(temp_dir, 'test_output.pkl')
        restart_filename = os.path.join(temp_dir, 'test_restart.pkl')
        status_filename = os.path.join(temp_dir, 'test_status')
        progress_filename = os.path.join(temp_dir, 'test_progress.out')
        mechanisms_path = Path(__file__).parent.parent.parent / "src" / "multipart" / "mechanisms"
        species_data_path = Path(__file__).parent.parent.parent / "src" / "multipart" / "species_data"

        # Test that function can be called without errors (smoke test)
        # Note: This will actually run a simulation, so we use minimal parameters
        # try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)
            result = simulate_parcel(
                aero_spec_names=aero_spec_names,
                aero_spec_masses=aero_spec_masses,
                num_concs=num_concs,
                pHs=pHs,
                z_start=0.0,
                z_end=2.0,  # Very short simulation
                dt=1.0,
                updraft_velocity=1.0,
                S0=0.85,
                P0=101325,
                T0=298,
                accom=1.0,
                output_filename=output_filename,
                restart_filename=restart_filename,
                status_filename=status_filename,
                progress_filename=progress_filename,
                print_to_screen=False,  # Suppress output
                condensation=True,
                cocondensation=False,
                aq_chemistry=None,
                gas_chemistry=False,
                mechanism_data_path=str(mechanisms_path) + "/",
                specdata_path=str(species_data_path) + "/"
            )

        # If we get here without exception, basic functionality works
        assert True

def test_simulate_parcel_gas_chem():
    """Test simulate_parcel with basic parameters."""
    # Set up minimal test data
    aero_spec_names = np.array([["SO4", "H2O"]])
    aero_spec_masses = np.array([[1e-15, 0.0]])  # Very small masses
    num_concs = np.array([1e6])
    pHs = np.array([5.0])

    # Use temporary directory for output files
    with tempfile.TemporaryDirectory() as temp_dir:
        output_filename = os.path.join(temp_dir, 'test_output.pkl')
        restart_filename = os.path.join(temp_dir, 'test_restart.pkl')
        status_filename = os.path.join(temp_dir, 'test_status')
        progress_filename = os.path.join(temp_dir, 'test_progress.out')
        mechanisms_path = Path(__file__).parent.parent.parent / "src" / "multipart" / "mechanisms"
        species_data_path = Path(__file__).parent.parent.parent / "src" / "multipart" / "species_data"

        # Test that function can be called without errors (smoke test)
        # Note: This will actually run a simulation, so we use minimal parameters
        # try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="unsafe cast from int64 to int32", category=Warning)
            result = simulate_parcel(
                aero_spec_names=aero_spec_names,
                aero_spec_masses=aero_spec_masses,
                num_concs=num_concs,
                pHs=pHs,
                z_start=0.0,
                z_end=2.0,  # Very short simulation
                dt=1.0,
                updraft_velocity=1.0,
                S0=0.85,
                P0=101325,
                T0=298,
                accom=1.0,
                output_filename=output_filename,
                restart_filename=restart_filename,
                status_filename=status_filename,
                progress_filename=progress_filename,
                print_to_screen=False,  # Suppress output
                condensation=True,
                cocondensation=False,
                aq_chemistry=None,
                gas_chemistry=True,
                mechanism_data_path=str(mechanisms_path) + "/",
                specdata_path=str(species_data_path) + "/"
            )

        # If we get here without exception, basic functionality works
        assert True

def test_simulate_parcel_cocondensation():
    """Test simulate_parcel with basic parameters."""
    # Set up minimal test data
    aero_spec_names = np.array([["SO4", "H2O"]])
    aero_spec_masses = np.array([[1e-15, 0.0]])  # Very small masses
    num_concs = np.array([1e6])
    pHs = np.array([5.0])

    # Use temporary directory for output files
    with tempfile.TemporaryDirectory() as temp_dir:
        output_filename = os.path.join(temp_dir, 'test_output.pkl')
        restart_filename = os.path.join(temp_dir, 'test_restart.pkl')
        status_filename = os.path.join(temp_dir, 'test_status')
        progress_filename = os.path.join(temp_dir, 'test_progress.out')
        mechanisms_path = Path(__file__).parent.parent.parent / "src" / "multipart" / "mechanisms"
        species_data_path = Path(__file__).parent.parent.parent / "src" / "multipart" / "species_data"

        # Test that function can be called without errors (smoke test)
        # Note: This will actually run a simulation, so we use minimal parameters
        # try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)
            result = simulate_parcel(
                aero_spec_names=aero_spec_names,
                aero_spec_masses=aero_spec_masses,
                num_concs=num_concs,
                pHs=pHs,
                z_start=0.0,
                z_end=2.0,  # Very short simulation
                dt=1.0,
                updraft_velocity=1.0,
                S0=0.85,
                P0=101325,
                T0=298,
                accom=1.0,
                output_filename=output_filename,
                restart_filename=restart_filename,
                status_filename=status_filename,
                progress_filename=progress_filename,
                print_to_screen=False,  # Suppress output
                condensation=True,
                cocondensation=True,
                aq_chemistry=None,
                gas_chemistry=False,
                gas_names=['SO2'],
                gas_concs=[1.0],
                mechanism_data_path=str(mechanisms_path) + "/",
                specdata_path=str(species_data_path) + "/"
            )

        # If we get here without exception, basic functionality works
        assert True

def test_simulate_parcel_aqueous_chem():
    """Test simulate_parcel with basic parameters."""
    # Set up minimal test data
    aero_spec_names = np.array([["SO4", "H2O"]])
    aero_spec_masses = np.array([[1e-15, 0.0]])  # Very small masses
    num_concs = np.array([1e6])
    pHs = np.array([5.0])

    # Use temporary directory for output files
    with tempfile.TemporaryDirectory() as temp_dir:
        output_filename = os.path.join(temp_dir, 'test_output.pkl')
        restart_filename = os.path.join(temp_dir, 'test_restart.pkl')
        status_filename = os.path.join(temp_dir, 'test_status')
        progress_filename = os.path.join(temp_dir, 'test_progress.out')
        mechanisms_path = Path(__file__).parent.parent.parent / "src" / "multipart" / "mechanisms"
        species_data_path = Path(__file__).parent.parent.parent / "src" / "multipart" / "species_data"

        # Test that function can be called without errors (smoke test)
        # Note: This will actually run a simulation, so we use minimal parameters
        # try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)
            result = simulate_parcel(
                aero_spec_names=aero_spec_names,
                aero_spec_masses=aero_spec_masses,
                num_concs=num_concs,
                pHs=pHs,
                z_start=0.0,
                z_end=2.0,  # Very short simulation
                dt=1.0,
                updraft_velocity=1.0,
                S0=0.85,
                P0=101325,
                T0=298,
                accom=1.0,
                output_filename=output_filename,
                restart_filename=restart_filename,
                status_filename=status_filename,
                progress_filename=progress_filename,
                print_to_screen=False,  # Suppress output
                condensation=True,
                cocondensation=True,
                aq_chemistry=['sulfate'],
                gas_chemistry=False,
                gas_names=None,
                gas_concs=None,
                mechanism_data_path=str(mechanisms_path) + "/",
                specdata_path=str(species_data_path) + "/"
            )

        # If we get here without exception, basic functionality works
        assert True

def test_simulate_parcel_path_warnings():
    """Test that simulate_parcel issues warnings for missing paths."""
    aero_spec_names = np.array([["SO4", "H2O"]])
    aero_spec_masses = np.array([[1e-25, 0.0]])
    num_concs = np.array([1e6])
    pHs = np.array([5.0])

    with tempfile.TemporaryDirectory() as temp_dir:
        output_filename = os.path.join(temp_dir, 'test_output.pkl')

        # Test warning for mechanism_data_path
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                simulate_parcel(
                    aero_spec_names=aero_spec_names,
                    aero_spec_masses=aero_spec_masses,
                    num_concs=num_concs,
                    pHs=pHs,
                    z_end=1.0,  # Very short
                    output_filename=output_filename,
                    print_to_screen=False,
                    mechanism_data_path=None,  # Should trigger warning
                    specdata_path=None  # Should trigger warning
                )
            except Exception:
                pass  # Expected to fail, we just want to check warnings

            # Check that warnings were issued
            warning_messages = [str(warning.message) for warning in w]
            assert any("mechanism path" in msg.lower() for msg in warning_messages)
            assert any("species data path" in msg.lower() for msg in warning_messages)

def test_simulate_les_basic_parameters():
    """Test simulate_parcel with basic parameters."""
    # Set up minimal test data
    aero_spec_names = np.array([["SO4", "H2O"]])
    aero_spec_masses = np.array([[1e-25, 0.0]])  # Very small masses
    num_concs = np.array([1e6])
    pHs = np.array([5.0])
    trajectory_data={}
    trajectory_data['t']=np.array([1.0,2.0,3.0])
    trajectory_data['x']=np.zeros(len(trajectory_data['t']))
    trajectory_data['y']=np.zeros(len(trajectory_data['t']))
    trajectory_data['z']=np.repeat(100, len(trajectory_data['t']))
    trajectory_data['T']=np.repeat(298, len(trajectory_data['t']))
    trajectory_data['P']=np.repeat(101325, len(trajectory_data['t']))
    trajectory_data['s']=np.repeat(0.85, len(trajectory_data['t']))
    trajectory_data['gas']=None

    # Use temporary directory for output files
    with tempfile.TemporaryDirectory() as temp_dir:
        output_filename = os.path.join(temp_dir, 'test_output.pkl')
        restart_filename = os.path.join(temp_dir, 'test_restart.pkl')
        status_filename = os.path.join(temp_dir, 'test_status')
        progress_filename = os.path.join(temp_dir, 'test_progress.out')
        mechanisms_path = Path(__file__).parent.parent.parent / "src" / "multipart" / "mechanisms"
        species_data_path = Path(__file__).parent.parent.parent / "src" / "multipart" / "species_data"

        # Test that function can be called without errors (smoke test)
        # Note: This will actually run a simulation, so we use minimal parameters
        # try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)
            simulate_les_trajectory(
                aero_spec_names, aero_spec_masses, num_concs, pHs, trajectory_data,
                dt=1.0, restart_filename=restart_filename, radius_scale='log',
                output_filename=output_filename, aq_chemistry=None,
                write_every=1.0, condensation=True, gas_chemistry=False, print_to_screen=False,
                cocondensation=False, relaxation_time=0.0, mechanism_data_path=str(mechanisms_path) + "/",
                specdata_path=str(species_data_path) + "/")

        # If we get here without exception, basic functionality works
        assert True

def test_simulate_les_gas_chem():
    """Test simulate_parcel with basic parameters."""
    # Set up minimal test data
    aero_spec_names = np.array([["SO4", "H2O"]])
    aero_spec_masses = np.array([[1e-25, 0.0]])  # Very small masses
    num_concs = np.array([1e6])
    pHs = np.array([5.0])
    trajectory_data={}
    trajectory_data['t']=np.array([1.0,2.0,3.0])
    trajectory_data['x']=np.zeros(len(trajectory_data['t']))
    trajectory_data['y']=np.zeros(len(trajectory_data['t']))
    trajectory_data['z']=np.repeat(100, len(trajectory_data['t']))
    trajectory_data['T']=np.repeat(298, len(trajectory_data['t']))
    trajectory_data['P']=np.repeat(101325, len(trajectory_data['t']))
    trajectory_data['s']=np.repeat(0.85, len(trajectory_data['t']))
    trajectory_data['gas']={}

    # Use temporary directory for output files
    with tempfile.TemporaryDirectory() as temp_dir:
        output_filename = os.path.join(temp_dir, 'test_output.pkl')
        restart_filename = os.path.join(temp_dir, 'test_restart.pkl')
        status_filename = os.path.join(temp_dir, 'test_status')
        progress_filename = os.path.join(temp_dir, 'test_progress.out')
        mechanisms_path = Path(__file__).parent.parent.parent / "src" / "multipart" / "mechanisms"
        species_data_path = Path(__file__).parent.parent.parent / "src" / "multipart" / "species_data"

        # Test that function can be called without errors (smoke test)
        # Note: This will actually run a simulation, so we use minimal parameters
        # try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="unsafe cast from int64 to int32", category=Warning)
            simulate_les_trajectory(
                aero_spec_names, aero_spec_masses, num_concs, pHs, trajectory_data,
                dt=1.0, restart_filename=restart_filename, radius_scale='log',
                output_filename=output_filename, aq_chemistry=None,
                write_every=1.0, condensation=True, gas_chemistry=True, print_to_screen=False,
                cocondensation=False, relaxation_time=0.0, mechanism_data_path=str(mechanisms_path) + "/",
                specdata_path=str(species_data_path) + "/")

        # If we get here without exception, basic functionality works
        assert True

def test_simulate_les_cocondensation():
    """Test simulate_parcel with basic parameters."""
    # Set up minimal test data
    aero_spec_names = np.array([["SO4", "H2O"]])
    aero_spec_masses = np.array([[1e-25, 0.0]])  # Very small masses
    num_concs = np.array([1e6])
    pHs = np.array([5.0])
    trajectory_data={}
    trajectory_data['t']=np.array([1.0,2.0,3.0])
    trajectory_data['x']=np.zeros(len(trajectory_data['t']))
    trajectory_data['y']=np.zeros(len(trajectory_data['t']))
    trajectory_data['z']=np.repeat(100, len(trajectory_data['t']))
    trajectory_data['T']=np.repeat(298, len(trajectory_data['t']))
    trajectory_data['P']=np.repeat(101325, len(trajectory_data['t']))
    trajectory_data['s']=np.repeat(0.85, len(trajectory_data['t']))
    trajectory_data['gas']={'SO2': np.repeat(1.0, len(trajectory_data['t']))}

    # Use temporary directory for output files
    with tempfile.TemporaryDirectory() as temp_dir:
        output_filename = os.path.join(temp_dir, 'test_output.pkl')
        restart_filename = os.path.join(temp_dir, 'test_restart.pkl')
        status_filename = os.path.join(temp_dir, 'test_status')
        progress_filename = os.path.join(temp_dir, 'test_progress.out')
        mechanisms_path = Path(__file__).parent.parent.parent / "src" / "multipart" / "mechanisms"
        species_data_path = Path(__file__).parent.parent.parent / "src" / "multipart" / "species_data"

        # Test that function can be called without errors (smoke test)
        # Note: This will actually run a simulation, so we use minimal parameters
        # try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)
            simulate_les_trajectory(
                aero_spec_names, aero_spec_masses, num_concs, pHs, trajectory_data,
                dt=1.0, restart_filename=restart_filename, radius_scale='log',
                output_filename=output_filename, aq_chemistry=None,
                write_every=1.0, condensation=True, gas_chemistry=False, print_to_screen=False,
                cocondensation=True, relaxation_time=0.0, mechanism_data_path=str(mechanisms_path) + "/",
                specdata_path=str(species_data_path) + "/")

        # If we get here without exception, basic functionality works
        assert True

def test_simulate_les_aqueous_chem():
    """Test simulate_parcel with basic parameters."""
    # Set up minimal test data
    aero_spec_names = np.array([["OC", "H2O"]])
    aero_spec_masses = np.array([[1e-25, 0.0]])  # Very small masses
    num_concs = np.array([1e6])
    pHs = np.array([5.0])
    trajectory_data={}
    trajectory_data['t']=np.array([1.0,2.0,3.0])
    trajectory_data['x']=np.zeros(len(trajectory_data['t']))
    trajectory_data['y']=np.zeros(len(trajectory_data['t']))
    trajectory_data['z']=np.repeat(100, len(trajectory_data['t']))
    trajectory_data['T']=np.repeat(298, len(trajectory_data['t']))
    trajectory_data['P']=np.repeat(101325, len(trajectory_data['t']))
    trajectory_data['s']=np.repeat(0.85, len(trajectory_data['t']))
    trajectory_data['gas']={}

    # Use temporary directory for output files
    with tempfile.TemporaryDirectory() as temp_dir:
        output_filename = os.path.join(temp_dir, 'test_output.pkl')
        restart_filename = os.path.join(temp_dir, 'test_restart.pkl')
        status_filename = os.path.join(temp_dir, 'test_status')
        progress_filename = os.path.join(temp_dir, 'test_progress.out')
        mechanisms_path = Path(__file__).parent.parent.parent / "src" / "multipart" / "mechanisms"
        species_data_path = Path(__file__).parent.parent.parent / "src" / "multipart" / "species_data"

        # Test that function can be called without errors (smoke test)
        # Note: This will actually run a simulation, so we use minimal parameters
        # try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)
            simulate_les_trajectory(
                aero_spec_names, aero_spec_masses, num_concs, pHs, trajectory_data,
                dt=1.0, restart_filename=restart_filename, radius_scale='log',
                output_filename=output_filename, aq_chemistry=['sulfate'],
                write_every=1.0, condensation=True, gas_chemistry=False, print_to_screen=False,
                cocondensation=False, relaxation_time=0.0, mechanism_data_path=str(mechanisms_path) + "/",
                specdata_path=str(species_data_path) + "/")

        # If we get here without exception, basic functionality works
        assert True

def test_simulate_les_trajectory_path_warnings():
    """Test that simulate_les_trajectory issues warnings for missing paths."""
    aero_spec_names = np.array(["SO4"])
    aero_spec_masses = np.array([[1e-25]])
    num_concs = np.array([1e6])
    pHs = np.array([5.0])

    # Mock trajectory data - minimal structure
    trajectory_data = {
        'time': np.array([0.0, 1.0]),
        'z': np.array([0.0, 1.0]),
        'T': np.array([298.0, 298.0]),
        'P': np.array([101325.0, 101325.0]),
        'S': np.array([0.85, 0.85])
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        output_filename = os.path.join(temp_dir, 'test_output.pkl')

        # Test warning for mechanism_data_path
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                simulate_les_trajectory(
                    aero_spec_names=aero_spec_names,
                    aero_spec_masses=aero_spec_masses,
                    num_concs=num_concs,
                    pHs=pHs,
                    trajectory_data=trajectory_data,
                    output_filename=output_filename,
                    print_to_screen=False,
                    mechanism_data_path=None,  # Should trigger warning
                    specdata_path=None  # Should trigger warning
                )
            except Exception:
                pass  # Expected to fail, we just want to check warnings

            # Check that warnings were issued
            warning_messages = [str(warning.message) for warning in w]
            assert any("mechanism path" in msg.lower() for msg in warning_messages)
            assert any("species data path" in msg.lower() for msg in warning_messages)
