"""
Initialization for cosmicqc package
"""

from .analyze import find_outliers, identify_outliers, label_outliers
from .detection import PerinuclearSignalDetector

# note: version placeholder is updated during build
# by poetry-dynamic-versioning.
__version__ = "0.0.0"
