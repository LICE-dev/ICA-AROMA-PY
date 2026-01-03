# ICA-AROMA (Python 3)

Python 3 port and refactoring of **ICA-AROMA** (*Independent Component Analysis -- Automatic Removal Of Motion Artifacts*).

This package provides a modern Python implementation of ICA-AROMA, distributed via **PyPI** and intended to be used primarily as a **Python library**, with optional command-line usage and optional **Nipype** integration.

> **Important**\
> This repository is **not a replacement for the original ICA-AROMA > project**.
> For the scientific background, theory, and full methodological description, please refer to the official ICA-AROMA repository and manual:
>
> https://github.com/maartenmennes/ICA-AROMA
> https://github.com/maartenmennes/ICA-AROMA/blob/master/Manual.pdf

------------------------------------------------------------------------

## Overview

ICA-AROMA is a data-driven method for identifying and removing motion-related independent components from fMRI data.

This Python 3 refactoring focuses on:

-   modern Python compatibility
-   modular and maintainable code
-   optional heavy dependencies
-   easy integration into larger pipelines (e.g. SWANe)

No methodological changes are introduced compared to the original implementation.

------------------------------------------------------------------------

## Installation

The package is distributed via **PyPI**.

``` bash
pip install ica-aroma
```

It is strongly recommended to install the package inside a virtual environment.

------------------------------------------------------------------------

## Optional Dependencies

### Nipype

Support for executing ICA-AROMA as part of a **Nipype workflow** is optional.

Install Nipype only if this functionality is required:

``` bash
pip install nipype
```

If Nipype-related features are requested without Nipype being installed, the package will stop execution with a **clear, user-oriented error message** explaining how to enable this functionality.

------------------------------------------------------------------------

## Usage

### As a Python library

The primary intended usage is as a Python module inside a larger analysis pipeline.

``` python
from ica_aroma.pipeline import run_aroma

run_aroma(
    in_file="input_func.nii.gz",
    out_dir="output_dir",
    motion_parameters="motion.txt",
)
```

This executes ICA-AROMA using the direct Python implementation, without Nipype.

------------------------------------------------------------------------

### Nipype-based execution

When Nipype is installed, ICA-AROMA can be executed through a Nipype workflow.

``` python
from ica_aroma.pipeline import run_aroma

run_aroma(
    in_file="input_func.nii.gz",
    out_dir="output_dir",
    use_nipype=True,
)
```

------------------------------------------------------------------------

## Command Line Interface

Although primarily designed as a library, a command-line entry point is provided for convenience and testing.

``` bash
ica-aroma --help
```

The CLI performs validation of:

-   required arguments
-   incompatible options
-   missing optional dependencies (e.g. Nipype)

All errors are reported with **actionable and explicit messages**.

------------------------------------------------------------------------

## Error Handling Philosophy

-   No raw `ModuleNotFoundError` is exposed to the user
-   Optional features fail gracefully
-   Error messages explicitly describe:
    -   the requested feature
    -   the missing dependency
    -   the steps required to enable it

------------------------------------------------------------------------

## Project Structure

``` text
ica_aroma/
├── cli.py
├── pipeline.py
├── ICA_AROMA_functions.py
├── __init__.py
```

-   `ICA_AROMA_functions.py`\
    Core ICA-AROMA algorithmic logic

-   `pipeline.py`\
    Execution logic (direct Python or Nipype-based)

-   `cli.py`\
    Argument parsing and runtime validation

------------------------------------------------------------------------

## Relationship to the Original ICA-AROMA Project

This repository is a **Python 3 refactoring and packaging effort**.

-   Algorithmic logic and scientific concepts are derived from the original work
-   No changes to the underlying method are introduced
-   Users should always cite and refer to the original ICA-AROMA publication

Official repository and manual:

-   https://github.com/maartenmennes/ICA-AROMA
-   https://github.com/maartenmennes/ICA-AROMA/blob/master/Manual.pdf

------------------------------------------------------------------------

## License

This project follows the license of the original ICA-AROMA project unless explicitly stated otherwise.

Please refer to the original repository for licensing and citation details.

------------------------------------------------------------------------

## Acknowledgements

-   Original ICA-AROMA authors
-   Nipype developers
-   SWANe project contributors