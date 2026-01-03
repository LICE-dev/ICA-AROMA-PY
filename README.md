<h1 align="center"> ICA-AROMA (Python 3)</h1><br>

<h3 align="center">
Independent Component Analysis – Automatic Removal Of Motion Artifacts<br>
Python 3 refactoring and packaging
</h3>

---

## Table of Contents

- [Introduction](#introduction)
- [Scope](#scope)
- [Installation](#installation)
- [Usage](#usage)
- [Optional Dependencies](#optional-dependencies)
- [Documentation](#documentation)
- [Relationship to the Original Project](#relationship-to-the-original-project)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Introduction

**ICA-AROMA** is a data-driven method for identifying and removing motion-related
independent components from fMRI data.

This repository provides a **Python 3 refactoring and packaging** of ICA-AROMA,
distributed via **PyPI** and intended to be used as a **Python library** inside
larger neuroimaging pipelines (e.g. SWANe).

No changes are introduced to the original algorithm or methodology.

---

## Scope

This project aims to:

- provide a clean and modular Python 3 implementation
- allow usage without mandatory heavy dependencies
- support optional execution through **Nipype**
- simplify integration into automated workflows

This project **does not aim** to:
- replace the original ICA-AROMA implementation
- modify or extend the original method
- provide a full neuroimaging platform or GUI

---

## Installation

The package is distributed via **PyPI**.

```bash
pip install ica-aroma
```

It is recommended to install the package inside a virtual environment.

---

## Usage

### As a Python library

The primary intended usage is programmatic, inside a Python pipeline.

```python
from ica_aroma.pipeline import run_aroma

run_aroma(
    in_file="input_func.nii.gz",
    out_dir="output_dir",
    motion_parameters="motion.txt",
)
```

This executes ICA-AROMA using the direct Python implementation.

---

### Command Line Interface

A command-line entry point is provided mainly for convenience and testing.

```bash
ica-aroma --help
```

The CLI validates arguments and reports clear, actionable error messages.

---

## Optional Dependencies

### Nipype

Support for execution through a **Nipype workflow** is optional.

```bash
pip install nipype
```

If Nipype-related features are requested without Nipype being installed, execution
will stop with a **human-readable error message** explaining how to enable this
functionality.

---

## Documentation

For the scientific background, theory, and full methodological description,
please refer to the **official ICA-AROMA repository and manual**:

- https://github.com/maartenmennes/ICA-AROMA

This repository focuses exclusively on implementation and packaging aspects.

---

## Relationship to the Original Project

This repository is a **Python 3 refactoring and redistribution** effort.

- Algorithmic logic derives from the original ICA-AROMA project
- No methodological changes are introduced
- Users should always cite the original ICA-AROMA publication

Original project and documentation:

- https://github.com/maartenmennes/ICA-AROMA

---

## License

This project follows the license of the original ICA-AROMA project unless
explicitly stated otherwise.

Refer to the original repository for licensing and citation details.

---

## Acknowledgements

- Original ICA-AROMA authors
- Nipype developers
- SWANe project contributors