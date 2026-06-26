# Legacy HISCALE example

This directory preserves the original HISCALE example script as source material for the refactored HISCALE April 25, 2016 case.

The public entry point for the case is now:

```text
../run_case.py
```

The preprocessing logic is being migrated into:

```text
../helpers/preprocess_inputs.py
```

This legacy directory may be removed after the refactored preprocessing workflow has been verified.

The files in this directory are preserved for reference and may not use the final refactored paths.
