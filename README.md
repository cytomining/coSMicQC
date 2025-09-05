<img height="200" src="https://raw.githubusercontent.com/cytomining/coSMicQC/main/media/logo/with-text-for-light-bg.png?raw=true">

# Single cell Morphology Quality Control

![PyPI - Version](https://img.shields.io/pypi/v/cosmicqc)
[![Build Status](https://github.com/cytomining/coSMicQC/actions/workflows/run-tests.yml/badge.svg?branch=main)](https://github.com/cytomining/coSMicQC/actions/workflows/run-tests.yml?query=branch%3Amain)
![Coverage Status](https://raw.githubusercontent.com/cytomining/coSMicQC/main/media/coverage-badge.svg)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Software DOI badge](https://zenodo.org/badge/DOI/10.5281/zenodo.14797008.svg)](https://doi.org/10.5281/zenodo.14797008)

> 🌠 Navigate the cosmos of single-cell morphology with confidence — coSMicQC keeps your data on course!

coSMicQC is a Python package to evaluate converted single-cell morphology outputs from CytoTable.

It can be challenging to get "perfect" single-cell compartment segmentation across large high-throughput screens when performing object detection in CellProfiler (or similar software).
Technical artifacts can arise during segmentation, leading to issues such as under-segmentation, over-segmentation, or the erroneous segmentation of background noise, smudges, or bright artifacts.

In single-cell analysis, intriguing phenotypes often emerge by examining morphological differences.
However, technical outliers in the data can obscure these biological insights, compromising the validity of the findings.

By utilizing specific morphological features extracted with CellProfiler, particularly AreaShape features, you can identify technically incorrect segmentations.
These can then be labeled or removed before further preprocessing steps, such as those performed with pycytominer.

> 🌟 Check out our [blog post](https://waysciencelab.com/2024/12/20/cosmicqc.html) on this for a greater understanding of the background and how coSMicQC can help you!

## Installation

Install coSMicQC from [PyPI](https://pypi.org/project/coSMicQC/) or from source:

```shell
# install from pypi
pip install coSMicQC

# install directly from source
pip install git+https://github.com/cytomining/coSMicQC.git
```

## Contributing, Development, and Testing

Please see our [contributing](https://cytomining.github.io/coSMicQC/main/contributing) documentation for more details on contributions, development, and testing.

## References

- [pycytominer](https://github.com/cytomining/pycytominer)
- [CellProfiler](https://github.com/CellProfiler/CellProfiler)
- [CytoTable](https://github.com/cytomining/CytoTable)
