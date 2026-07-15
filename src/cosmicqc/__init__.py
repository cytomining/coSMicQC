"""
Initialization for cosmicqc package
"""

from importlib.metadata import PackageNotFoundError, version

from .analyze import find_outliers, identify_outliers, label_outliers
from .detection import ContaminationDetector

# version is derived from installed package metadata, which is populated at
# build time by setuptools-scm from the project's git history.
try:
    __version__ = version("coSMicQC")
except PackageNotFoundError:
    __version__ = "0.0.0"
