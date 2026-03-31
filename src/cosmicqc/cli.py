"""
Setup coSMicQC CLI through python-fire
"""

import inspect
import sys
import types
from typing import Optional

import fire
from fire import helptext, inspectutils, value_types
from fire.core import Display, FireError, _DictAsString, _OneLineResult
from fire.trace import FireTrace

from . import analyze


# used to avoid bugs with python-fire and pandas string-based repr compatibility.
# referenced from https://github.com/google/python-fire/pull/446
# to be removed after python-fire merges changes (uncertain of timeline)
def HasCustomRepr(component: object) -> bool:
    """Reproduces above HasCustomStr function to determine if component has a
    custom __repr__ method.

    ...

    Args:
      component: The object to check for a custom __repr__ method.
    Returns:
      Whether `component` has a custom __repr__ method.
    """
    if hasattr(component, "__repr__"):
        class_attrs = inspectutils.GetClassAttrsDict(type(component)) or {}
        repr_attr = class_attrs.get("__repr__")
        if repr_attr and repr_attr.defining_class is not object:
            return True
    return False


# used to avoid bugs with python-fire and pandas string-based repr compatibility.
# referenced with modifications from https://github.com/google/python-fire/pull/446
# to be removed after python-fire merges changes (uncertain of timeline)
# ignore rule below added to help avoid triggering ruff linting checks on temporary fix.
def _PrintResult(  # noqa: C901
    component_trace: FireTrace, verbose: bool = False, serialize: Optional[bool] = None
) -> None:
    """Prints the result of the Fire call to stdout in a human readable way."""
    result = component_trace.GetResult()
    # Allow users to modify the return value of the component and provide
    # custom formatting.
    if serialize:
        if not callable(serialize):
            raise FireError(
                "The argument `serialize` must be empty or callable:", serialize
            )
        result = serialize(result)
    if value_types.HasCustomStr(result):
        # If the object has a custom __str__ method, rather than one inherited from
        # object, then we use that to serialize the object.
        print(str(result))
        return

    elif HasCustomRepr(result):
        # Same as above, but for __repr__.
        # For pandas.DataFrame, __str__ is inherited from object, but __repr__ has
        # a custom implementation (see pandas.core.frame.DataFrame.__repr__)
        print(str(result))
        return

    if isinstance(result, (list, set, frozenset, types.GeneratorType)):
        for i in result:
            print(_OneLineResult(i))
    elif inspect.isgeneratorfunction(result):
        raise NotImplementedError
    elif isinstance(result, dict) and value_types.IsSimpleGroup(result):
        print(_DictAsString(result, verbose))
    elif isinstance(result, tuple):
        print(_OneLineResult(result))
    elif isinstance(result, value_types.VALUE_TYPES):
        if result is not None:
            print(result)
    else:
        help_text = helptext.HelpText(result, trace=component_trace, verbose=verbose)
        output = [help_text]
        # used for displaying output through python-fire
        Display(output, out=sys.stdout)


# replace the _PrintResult function with a fix for pandas dataframes
fire.core._PrintResult = _PrintResult


def cli_analyze() -> None:
    """
    Run the analyze module functions through python-fire CLI

    This function serves as the CLI entry point for functions
    within the analyze module.
    """

    fire.Fire(analyze)


if __name__ == "__main__":
    """
    Setup the CLI with python-fire for the coSMicQC package.

    This enables running the functions find_outliers and label_outliers
    from the command line interface through analyze.
    """

    cli_analyze()
