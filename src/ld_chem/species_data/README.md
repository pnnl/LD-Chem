# Species Data

This directory contains the physical and chemical properties for aerosol and gas species used in LD-Chem simulations.

## Contents

### `aero_data.dat` - Aerosol Species Properties

Contains properties for aerosol and aqueous-phase species that can exist in particles or droplets.

**Columns:**

| Column | Description | Unit |
|--------|-------------|------|
| `name` | Chemical formula or species identifier | - |
| `dens` | Bulk density | kg/m³ |
| `ions in soln` | Number of ions produced when dissolved (used for Raoult effect) | dimensionless |
| `molec wght` | Molecular weight | kg/mol |
| `kappa` | Hygroscopicity parameter ($\kappa$-Köhler theory) | dimensionless |

**Default aerosol species** (density > 0):
- **Inorganic ions**: SO₄, NO₃, Cl, NH₄, Na, Ca, CO₃
- **Organic compounds**: ARO1, ARO2, ALK1, OLE1, API1, API2, LIM1, LIM2, MSA, OC (organic carbon), BC (black carbon)
- **Other**: OIN (mineral dust)

**Default aqueous/dissolved species** (density = 0):
- Species that exist primarily in the aqueous phase or as dissolved gases
- SO₂, HSO₃, SO₃, H₂SO₄, HSO₄, O₃, H⁺, HNO₃, IEPOX, H₂O

### `gas_data.dat` - Gas-Phase Species Properties

Contains properties for gas-phase species available for reactions and/or aqueous partitioning.

**Columns:**

| Column | Description | Unit |
|--------|-------------|------|
| `name` | Chemical formula or species identifier | - |
| `alpha` | Mass accommodation coefficient | dimensionless |
| `molec wght` | Molecular weight | kg/mol |
| `H0` | Henry's law constant at 298 K | M/atm |
| `H_exp` | Temperature dependence parameter for Henry's law | K |

**Henry's Law Temperature Dependence:**

$$H(T) = H_0 \cdot \exp\left(H_{exp} \left(\frac{1}{T} - \frac{1}{298}\right)\right)$$


## Usage

### Reading Species Data

LD-Chem automatically reads these files during simulation initialization. The `specdata_path` argument in simulation functions controls which species data directory to use:

```python
from ld_chem import simulate_parcel

# Use default species data
simulate_parcel(
    aero_spec_names, aero_spec_masses, num_concs, pHs,
    output_filename='trajectory.pkl'
)

# Use custom species data directory
simulate_parcel(
    aero_spec_names, aero_spec_masses, num_concs, pHs,
    specdata_path='/path/to/custom/species_data/',
    output_filename='trajectory.pkl'
)
```

### Adding New Species

To add a new species:

1. **For aerosol species**, add a line to `aero_data.dat`:
   ```
   NEWSPEC        1500                0                   120d-3     0.2
   ```
   Ensure all four properties are provided.

2. **For gas species**, add a line to `gas_data.dat`:
   ```
   NEWGAS         0.05                 46d-3                   1.0d1      2.0d3
   ```
   Ensure all five properties are provided.

3. Reference the species by name in your simulation setup (e.g., in `aero_spec_names` or `gas_names`).

### Modifying Existing Species

Edit the corresponding `.dat` file directly. Common modifications include:

- **Hygroscopicity (κ)**: Adjust to change activation threshold
- **Molecular weight**: Update if correcting chemical formulas
- **Henry's law constant**: Adjust to change partitioning behavior
- **Accommodation coefficient**: Modify to change condensation kinetics

## File Format

Both files follow a simple tabular format:
- Header row begins with `#` and describes columns
- One species per line
- Whitespace-separated columns
- Scientific notation supported (e.g., `1.4d0` = 1.4 × 10⁰)

## Tips

- Keep species names consistent between `aero_data.dat` and chemical mechanism files
- Non-zero density in `aero_data.dat` indicates aerosol species
- Zero density in `aero_data.dat` indicates dissolved/aqueous species
- H0 values of 0 in `gas_data.dat` indicate non-condensable species
- Test modifications locally before using in production simulations
