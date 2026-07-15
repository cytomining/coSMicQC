<img height="200" src="https://raw.githubusercontent.com/cytomining/coSMicQC/main/media/logo/with-text-for-light-bg.png?raw=true">

# Single cell Morphology Quality Control

![PyPI - Version](https://img.shields.io/pypi/v/cosmicqc)
[![Build Status](https://github.com/cytomining/coSMicQC/actions/workflows/run-tests.yml/badge.svg?branch=main)](https://github.com/cytomining/coSMicQC/actions/workflows/run-tests.yml?query=branch%3Amain)
![Coverage Status](https://raw.githubusercontent.com/cytomining/coSMicQC/main/media/coverage-badge.svg)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Preprint DOI badge](https://img.shields.io/badge/Preprint_DOI-10.1101/2025.10.14.682427-blue)](https://doi.org/10.1101/2025.10.14.682427)
[![Software DOI badge](https://img.shields.io/badge/Software_DOI-10.5281/zenodo.14888111-blue)](https://doi.org/10.5281/zenodo.14797008)

> 🌠 Navigate the cosmos of single-cell morphology with confidence — coSMicQC keeps your data on course!

## Contents

<!--ts-->
* [Single cell Morphology Quality Control](#single-cell-morphology-quality-control)
   * [Contents](#contents)
   * [Overview](#overview)
   * [Repo Contents](#repo-contents)
   * [System Requirements](#system-requirements)
   * [Installation Guide](#installation-guide)
      * [Stable release (PyPI)](#stable-release-pypi)
      * [Development version (from source)](#development-version-from-source)
   * [Demo](#demo)
   * [Results](#results)
   * [License](#license)
   * [Issues](#issues)
   * [Citation](#citation)


<!--te-->

## Overview

coSMicQC is a Python package to evaluate converted single-cell morphology outputs from [CytoTable](https://github.com/cytomining/CytoTable).
Technical artifacts can arise during segmentation, leading to issues such as under-segmentation, over-segmentation, or the erroneous segmentation of background noise, smudges, or bright artifacts.
By utilizing specific morphological features extracted with CellProfiler, you can identify technically incorrect segmentations and label or remove them before downstream analysis.

Please confer with our docsite for more comprehensive information about the project: https://cytomining.github.io/coSMicQC/main/

> 🌟 Check out our [blog post](https://waysciencelab.com/2024/12/20/cosmicqc.html) for a deeper background and how coSMicQC can help.

## Repo Contents

- `src/cosmicqc`: coSMicQC source code.
- `tests`: test suite.
- `docs`: documentation sources and examples.
- `media`: project assets (e.g., coverage badge).
- `reports`: generated artifacts (figures, notebooks, or summaries).

## System Requirements

- Python: `>=3.10` (tested through 3.13).
- RAM/CPU: standard laptop/desktop is sufficient for typical plate-sized datasets; larger screens benefit from more RAM/cores for faster QC/plotting.
- OS: Linux, macOS, and Windows are supported via Python; CI tests run on GitHub Actions.

## Installation Guide

### Stable release (PyPI)

```shell
pip install coSMicQC
```

### Development version (from source)

```shell
pip install git+https://github.com/cytomining/coSMicQC.git
# or with uv for local development
uv sync
```

## Demo

- Examples and notebooks live under `docs/src/examples` and the published docs: https://cytomining.github.io/coSMicQC
- Quickstart blog post: https://waysciencelab.com/2024/12/20/cosmicqc.html

## Results

- Test and lint status: ![Build Status](https://github.com/cytomining/coSMicQC/actions/workflows/run-tests.yml/badge.svg?branch=main)
- Coverage: ![Coverage Status](https://raw.githubusercontent.com/cytomining/coSMicQC/main/media/coverage-badge.svg)

## License

BSD-3-Clause; see [LICENSE](LICENSE).

## Issues

Please open issues or feature requests at https://github.com/cytomining/coSMicQC/issues.

## Citation

If you use coSMicQC in your work, please cite:

- Software DOI: [10.5281/zenodo.14888111](https://doi.org/10.5281/zenodo.14888111)
- Preprint: [10.1101/2025.10.14.682427](https://doi.org/10.1101/2025.10.14.682427)
- Citation metadata: [CITATION.cff](CITATION.cff)
