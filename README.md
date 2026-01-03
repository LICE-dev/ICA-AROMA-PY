<h1 align="center">
    ICA-AROMA (Python 3)
</h1>
<br>
<h3 align="center">
    Independent Component Analysis – Automatic Removal Of Motion Artifacts
    <br>
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

**ICA-AROMA** is a data-driven method for identifying and removing motion-related independent components from fMRI data.

This repository provides a **Python 3 refactoring and packaging** of ICA-AROMA, distributed via **PyPI** and intended to be used as a **Python library** inside larger neuroimaging pipelines (e.g. [SWANe](https://github.com/LICE-dev/swane)).

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
pip install ica-aroma-py
```

It is recommended to install the package inside a virtual environment.

---

## Usage

ICA-AROMA can be executed either using the **direct Python engine** (default) or
through a **Nipype-based workflow**. Plot generation is enabled by default and
can be disabled explicitly if optional plotting dependencies are not installed.

---

### Direct execution (default engine)

This is the default execution mode and does not require Nipype.

```bash
ica-aroma -o out -i func.nii.gz -mc mc.par
```

Equivalent explicit form:

```bash
ica-aroma --engine direct -o out -i func.nii.gz -mc mc.par
```

---

### Nipype-based execution

To execute ICA-AROMA using a Nipype workflow, the execution engine must be set
explicitly to `nipype`.

```bash
ica-aroma --engine nipype -o out -i func.nii.gz -mc mc.par
```

Optional Nipype-specific arguments:

```bash
ica-aroma --engine nipype   -o out   -i func.nii.gz   -mc mc.par   --nprocs 12   --mp-context spawn
```

> [!WARNING]
> Nipype-based execution requires the optional dependency `nipype` to be
> installed:
>
> ```bash
> pip install ica-aroma-py[nipype]
> ```

---

### Plot generation (Matplotlib)

Plot generation is **enabled by default**.

If plotting dependencies (e.g. Matplotlib) are not installed, execution will
fail with a clear error message. To explicitly disable plot generation, use the
`-np / --noplots` flag.

```bash
ica-aroma -o out -i func.nii.gz -mc mc.par -np
```

This applies to both direct and Nipype execution modes.

```bash
ica-aroma --engine nipype -o out -i func.nii.gz -mc mc.par -np
```

To enable plotting support, install the optional plotting dependencies:

```bash
pip install ica-aroma-py[plots]
```

---

### FEAT mode

If a FEAT directory is available, ICA-AROMA can be executed using FEAT mode:

```bash
ica-aroma -o out -f feat_directory.feat
```

Additional optional arguments (e.g. `-den`, `-dim`, `-tr`) can be combined with
any execution mode as required.

---

### Notes

- The execution engine is selected using the `--engine` argument
  (`direct` or `nipype`).
- Plot generation is enabled by default and can be disabled using `-np`.
- Optional dependencies are validated at runtime with explicit error messages.
- Run `ica-aroma --help` to see the full list of available command-line options.

---

## Optional Dependencies

### Nipype

Support for execution through a **Nipype workflow** is optional.

```bash
pip install nipype
```

If Nipype-related features are requested without Nipype being installed, execution will stop with a **human-readable error message** explaining how to enable this functionality.

### Matplotlib

Matplotlib is required only for plotting and visualization features (e.g. feature distributions and diagnostic plots).

```bash
pip install matplotlib
```

If plotting-related features are requested without Matplotlib installed, execution will stop with a **human-readable error message** explaining how to enable this functionality.

---

## Documentation

For the scientific background, theory, and full methodological description, please refer to:

- the [**official ICA-AROMA repository**](https://github.com/maartenmennes/ICA-AROMA)
- the [**ICA-AROMA manual**](https://github.com/maartenmennes/ICA-AROMA/blob/master/Manual.pdf)

This repository focuses exclusively on implementation and packaging aspects.

---

## Relationship to the Original Project

This repository is a **Python 3 refactoring and redistribution** effort.

- Algorithmic logic derives from the original ICA-AROMA project
- No methodological changes are introduced
- Users should always cite the original ICA-AROMA publication

Original project and documentation:

- [**ICA-AROMA repository**](https://github.com/maartenmennes/ICA-AROMA)
- [**ICA-AROMA manual**](https://github.com/maartenmennes/ICA-AROMA/blob/master/Manual.pdf)

---

## License

This project follows the license of the original ICA-AROMA project unless explicitly stated otherwise.

Refer to the original repository for licensing and citation details.

---

## Acknowledgements

- Original ICA-AROMA authors
- Nipype developers
- [SWANe](https://github.com/LICE-dev/swane) project contributors