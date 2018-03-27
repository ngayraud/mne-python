# author: ngayraud
#
# Created on Wed Feb 21 11:13:50 2018.

import numpy as np
from ..utils import warn


def get_function(function):
    """Check function exists and return the callable.

    Returns a simple sinusoide if not found.

    Parameters
    ----------
    function : str | callable
        If str, mus be one of the known functions: 'sin', 'p300_target,
        'p300_nontarget'.
    """
    known_functions = {
        'sin': function_sin,
        'p300_target': function_p300_target,
        'p300_nontarget': function_p300_nontarget,
    }
    if isinstance(function, str) and function in known_functions.keys():
        return known_functions[function]
    elif hasattr(function, "__call__"):
        return function
    else:
        warn('unknown function. Sinusoide will be generated')
        return function_sin


def function_sin(times):
    """Generate a sinusoide waveform.    .

    Returns a simple sinusoide

    Parameters
    ----------
    times : array
        Array of times
    """
    return 1e-7 * np.sin(20 * np.pi * times)


def function_p300_target(times, peak=0.3, amplitude=15.0):
    """Generate a p300 target waveform.

    Create a p300 target waveform.

    Parameters
    ----------
    times : array
        Array of times
    peak : float
        peak of the p300
    amplitude : array
        amplitude of the p300
    """
    return (1e-9 * amplitude * np.cos(17. * (times - peak)) *
            np.exp(-(times - peak + 0.04)**2 / 0.02) +
            1e-9 * amplitude / 6.0 * np.sin(17. * (times - peak)) *
            np.exp(-(times - peak + 0.04 + 0.2)**2 / 0.02))


def function_p300_nontarget(times, peak=0.3, amplitude=15.0):
    """Generate a p300 nontarget waveform.

    Create a p300 nontarget waveform.

    Parameters
    ----------
    times : array
        Array of times
    peak : float
        peak of the p300
    amplitude : array
        amplitude of the p300
    """
    return (1e-9 * (amplitude / 4.0) * np.cos(17. * (times - peak)) *
            np.exp(-(times - peak + 0.04)**2 / 0.05) +
            1e-9 * (amplitude / 6.0) * np.sin(17. * (times - peak)) *
            np.exp(-(times - peak + 0.04 + 0.2)**2 / 0.02))
