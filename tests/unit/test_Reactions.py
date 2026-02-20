import pytest
import numpy as np
from pathlib import Path
from ld_chem.reactions import (
    AqReaction, GasReaction, AqueousReactions, GasReactions,
    make_AqReactions, make_GasReactions
)


def test_aq_reaction_creation():
    """Test AqReaction dataclass creation."""
    reaction = AqReaction(
        reactants=["SO2"],
        products=["HSO3", "H+"],
        rate0=5.0e5,
        neg_Ea_R=1960.0
    )
    assert reaction.reactants == ["SO2"]
    assert reaction.products == ["HSO3", "H+"]
    assert reaction.rate0 == 5.0e5
    assert reaction.neg_Ea_R == 1960.0


def test_aq_reaction_get_rate():
    """Test aqueous reaction rate calculation."""
    reaction = AqReaction(
        reactants=["SO2"],
        products=["HSO3", "H+"],
        rate0=5.0e5,
        neg_Ea_R=1960.0
    )
    T = 298  # 25°C
    rate = reaction.get_rate(T)

    # At reference temperature, should equal rate0
    assert np.isclose(rate, reaction.rate0)

    # Should change with temperature
    T_higher = 308  # 35°C
    rate_higher = reaction.get_rate(T_higher)
    assert rate_higher != rate


def test_gas_reaction_creation():
    """Test GasReaction dataclass creation."""
    reaction = GasReaction(
        reactants=["H2O2"],
        products=["OH", "OH"],
        rate0=7.66e-6,
        high_P_limit=0.0,
        T_dependence=0.0,
        form="exp"
    )
    assert reaction.reactants == ["H2O2"]
    assert reaction.products == ["OH", "OH"]
    assert reaction.rate0 == 7.66e-6
    assert reaction.high_P_limit == 0.0
    assert reaction.T_dependence == 0.0
    assert reaction.form == "exp"


def test_gas_reaction_get_rate_power():
    """Test gas reaction rate calculation with power law."""
    reaction = GasReaction(
        reactants=["A"],
        products=["B"],
        rate0=1e-10,
        high_P_limit=0.0,
        T_dependence=2.0,
        form="power"
    )
    S = 1.0
    T = 300.0
    P = 101325.0
    rate = reaction.get_rate(S, T, P)
    expected = 1e-10 * (T/300)**2.0
    assert np.isclose(rate, expected)


def test_gas_reaction_get_rate_exp():
    """Test gas reaction rate calculation with exponential."""
    reaction = GasReaction(
        reactants=["H2O2"],
        products=["OH", "OH"],
        rate0=7.66e-6,
        high_P_limit=0.0,
        T_dependence=0.0,
        form="exp"
    )
    S = 1.0
    T = 298.15
    P = 101325.0
    rate = reaction.get_rate(S, T, P)
    expected = 7.66e-6 * np.exp(0.0/T)
    assert np.isclose(rate, expected)


def test_aqueous_reactions_creation():
    """Test AqueousReactions dataclass creation."""
    reaction1 = AqReaction(["A", "B"], ["C"], 5.0e5, 1960.0)
    reaction2 = AqReaction(["D"], ["E", "F"], 5.0e5, 1500.0)
    reactions = (reaction1, reaction2)
    ids = (0, 1)
    aq_reactions = AqueousReactions(reactions=reactions, ids=ids)
    assert len(aq_reactions.reactions) == 2
    assert len(aq_reactions.ids) == 2
    assert aq_reactions.reactions[0].reactants == ["A", "B"]


def test_gas_reactions_creation():
    """Test GasReactions dataclass creation."""
    reaction1 = GasReaction(["A", "B"], ["C"], 7.66e-6, 0.0, 0.0, "exp")
    reaction2 = GasReaction(["D"], ["E", "F"], 3.55e-5, 0.0, 0.0, "exp")
    reactions = (reaction1, reaction2)
    ids = (0, 1)
    gas_reactions = GasReactions(reactions=reactions, ids=ids)
    assert len(gas_reactions.reactions) == 2
    assert len(gas_reactions.ids) == 2
    assert gas_reactions.reactions[0].reactants == ["A", "B"]


def test_make_aq_reactions():
    """Test creation of AqueousReactions from file."""
    # Path to mechanisms directory from test file location
    mechanisms_path = Path(__file__).parent.parent.parent / "src" / "ld_chem" / "mechanisms"

    chemistry = ["sulfate"]
    aq_reactions = make_AqReactions(chemistry=chemistry, mechanism_data_path=str(mechanisms_path) + "/")

    assert aq_reactions.reactions is not None
    assert aq_reactions.ids is not None
    assert len(aq_reactions.reactions) > 0
    assert len(aq_reactions.ids) > 0

    # Check that reactions have the expected structure
    for reaction in aq_reactions.reactions:
        assert isinstance(reaction, AqReaction)
        assert isinstance(reaction.reactants, list)
        assert isinstance(reaction.products, list)
        assert isinstance(reaction.rate0, float)
        assert isinstance(reaction.neg_Ea_R, float)


def test_make_gas_reactions():
    """Test creation of GasReactions from file."""
    mechanisms_path = Path(__file__).parent.parent.parent / "src" / "ld_chem" / "mechanisms"

    # make_GasReactions doesn't use chemistry parameter like make_AqReactions
    gas_reactions = make_GasReactions(mechanism_data_path=str(mechanisms_path) + "/")

    assert gas_reactions.reactions is not None
    assert gas_reactions.ids is not None
    assert len(gas_reactions.reactions) > 0
    assert len(gas_reactions.ids) > 0

    # Check that reactions have the expected structure
    for reaction in gas_reactions.reactions:
        assert isinstance(reaction, GasReaction)
        assert isinstance(reaction.reactants, list)
        assert isinstance(reaction.products, list)
        assert isinstance(reaction.rate0, float)
        assert isinstance(reaction.high_P_limit, float)
        assert isinstance(reaction.T_dependence, float)
        assert reaction.form in ["power", "exp", "troe", "HO2_water_enhancement"]
