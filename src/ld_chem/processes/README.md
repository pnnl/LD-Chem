# Processes

This directory contains the differential equation solvers and rate calculations for physical and chemical processes in LD-Chem simulations. Each module implements process-specific calculations using Numba for performance optimization.

## Contents

### `air_thermo.py` - Air Thermodynamics

Implements thermodynamic calculations for air parcel conditions.

**Key Functions:**
- `es(T)` - Saturation vapor pressure of water over a flat surface
- `dstate_dt(X0, V, dwc_dt)` - Calculates parcel state vector time derivatives
- `H2O_gas_conc(S, T, P)` - Converts saturation ratio to gas-phase water concentration
- `compute_thermo_props(T, P, S, V)` - Computes temperature, pressure, and other thermodynamic properties

**Purpose:**
- Tracks parcel thermodynamic evolution (temperature, pressure, saturation)
- Provides saturation properties for condensation calculations
- Uses Numba AOT (ahead-of-time) compilation for speed

### `water_uptake.py` - Water Condensation

Implements water vapor condensation kinetics using $\kappa$-Köhler theory.

**Key Functions:**
- `sigma_w(T)` - Surface tension of water as a function of temperature
- `ka(T, r, rho)` - Thermal conductivity of air
- `dv(T, r, P, accom)` - Water vapor diffusivity function
- `es(T)` - Saturation vapor pressure
- `dr_dt(radii, dry_radii, kappas, P, T, S, accom)` - Rate equation for water condensation

**References:**
- Based on pyrcel implementation: https://github.com/darothen/pyrcel

### `cocondensation.py` - Gas-aqueous mass transfer of semi-volatile gases

Implements co-condensation of condensable gases onto particles.

**Key Functions:**
- `dCaq_dt(X, radii, water_volumes, num_concs, molar_mass, alpha, Heff, T)` - Rate equation for gas-particle mass transfer
- `GasFeedback` - Data class tracking gas concentrations and rates for feedback effects

**Purpose:**
- Models condensation of condensable gases
- Solves coupled gas-particle equilibria
- Accounts for:
  - Henry's law solubility
  - Diffusional kinetics
  - Mass accommodation coefficients

**Parameters:**
- Gas accommodation coefficient (α): kinetic limitation factor
- Henry's law constant (H): solubility at equilibrium
- Molecular weight and diffusivity: transport properties

### `gas_chemistry.py` - Gas-Phase Chemistry

Implements gas-phase chemical reaction rate equations.

**Key Functions:**
- `dCgas_dt(Cgas_0, reactants_all, products_all, rates, gas_names, T, P)` - Rate equation for gas-phase species

**Purpose:**
- Calculates gas-phase chemical reaction rates
- Supports multiple reaction types:
  - Bimolecular reactions: A + B → products
  - Termolecular reactions: A + B + M → products
  - Temperature-dependent rate coefficients

**Temperature Dependence:**
- Power law: k(T) = k₀(T/300)^n
- Exponential: k(T) = k₀ exp(ΔE/T)
- Troe fall-off: Three-body recombination reactions
- Water-enhanced HO₂ self-reaction

### `aqueous_chemistry.py` - Aqueous-Phase Chemistry

Implements aqueous-phase chemical reaction rate equations for in-cloud processes.

**Key Functions:**
- `dCaq_dt(Caq_0, reactants_all, products_all, rates, aq_names, T)` - Rate equation for aqueous-phase species

**Purpose:**
- Calculates aqueous-phase reaction rates
- Implements pH-dependent chemistry (particularly for sulfur species)
- Handles complex equilibria:
  - SO₂/HSO₃/SO₃
  - H₂SO₄ dissociation
  - Ammonia/ammonium protonation
  - Nitrogen oxide equilibria

**Rate Laws:**
- Temperature-dependent Arrhenius rates
- pH-dependent rate modifications
- Aqueous-phase species concentrations (mol/m³)

## Implementation Details

### Numba Optimization

Most functions use Numba `@nb.njit()` decorators for:
- Just-in-time (JIT) compilation to machine code
- Significant speedup for numerical computations
- Python-compatible syntax with limited NumPy support

Ahead-of-time (AOT) compilation is used for:
- Initialization functions (with `auxcc.export`)
- Functions requiring C library linking (thermal conductivity, etc.)

### Solver Integration

These modules define `dX_dt` functions called by the main ODE solver in `systems.py`:
- Input: current state vector X
- Output: time derivatives dX/dt
- Used with scipy's `ode` integrators

## Usage

These modules are called automatically during simulation via the `Processes` class in `systems.py`. Users typically do not call these functions directly, but understanding them helps with:

- Troubleshooting simulation behavior
- Adding new chemical mechanisms
- Optimizing performance
- Modifying rate calculations


## References

Please refer to (xxx) for a complete list of references for the model formulation.
