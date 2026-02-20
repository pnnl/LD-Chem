import pytest
import numpy as np
import warnings
from ld_chem.systems import Processes, Feedbacks


def test_processes_creation():
    """Test Processes dataclass creation."""
    # Test default values
    processes = Processes()
    assert processes.condensation == True
    assert processes.cocondensation == False
    assert processes.aq_chemistry == False
    assert processes.gas_chemistry == False

    # Test custom values
    processes_custom = Processes(
        condensation=False,
        cocondensation=True,
        aq_chemistry=True,
        gas_chemistry=True
    )
    assert processes_custom.condensation == False
    assert processes_custom.cocondensation == True
    assert processes_custom.aq_chemistry == True
    assert processes_custom.gas_chemistry == True


def test_feedbacks_creation():
    """Test Feedbacks dataclass creation."""
    # Test default values
    feedbacks = Feedbacks()
    assert feedbacks.dwc_dt == 0.0
    assert feedbacks.dwv_dt == 0.0
    assert feedbacks.gases is None

    # Test custom values
    feedbacks_custom = Feedbacks(
        dwc_dt=1.5e-6,
        dwv_dt=-2.3e-7,
        gases=None
    )
    assert feedbacks_custom.dwc_dt == 1.5e-6
    assert feedbacks_custom.dwv_dt == -2.3e-7
