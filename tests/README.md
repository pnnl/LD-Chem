# Test Organization

This directory contains the test suite for the LD-Chem package, organized into unit and integration tests.

## Directory Structure

```
tests/
├── unit/               # Unit tests for functions and classes (12 files)
│   ├── test_air_themo.py
│   ├── test_aqueous_chemistry.py
│   ├── test_cocondensation.py
│   ├── test_constants.py
│   ├── test_gas_chemistry.py
│   ├── test_gases.py
│   ├── test_particles.py
│   ├── test_Reactions.py
│   ├── test_scenario.py
│   ├── test_systems.py
│   ├── test_utilities.py
│   └── test_water_uptake.py
└── integration/        # Integration tests (marked with @pytest.mark.integration)
    └── test_run.py
```

## Running Tests

### Unit Tests Only (Default)
```bash
pytest  # Runs all unit tests
```

### Integration Tests Only
```bash
pytest -m integration
```

### All Tests
```bash
pytest --runxfail  # Or modify pytest.ini to include integration tests
```

### Specific Test Categories
```bash
pytest tests/processes/     # Process unit tests
pytest tests/objects/       # Object unit tests
pytest tests/integration/   # Integration tests
```

## Test Classification

### Unit Tests
- Test individual functions/classes in isolation
- Use mocks/stubs for dependencies
- Fast execution
- Located in `tests/processes/` and `tests/objects/`

### Integration Tests
- Test interactions between components
- May require more setup and external dependencies
- Slower execution
- Marked with `@pytest.mark.integration`
- Located in `tests/integration/`
