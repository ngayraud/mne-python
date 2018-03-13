# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:13:50 2018

@author: ngayraud
"""
import numpy as np
from ..utils import warn

def get_function(function):
    """ Check function exist and return the callable. Returns sinusoide if
        not found.
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
    """Function to generate a sinusoide waveform"""
    return 1e-7 * np.sin(20 * np.pi * times)


def function_p300_target(times, peak=0.3, amplitude=15.0):
    """Function to generate a p300 target waveform"""
    return (1e-9 * amplitude * np.cos(17.*(times-peak)) *
            np.exp(-(times - peak+0.04)**2 / 0.02) +
            1e-9 * amplitude/6.0 * np.sin(17.*(times-peak)) *
            np.exp(-(times - peak+0.04+0.2)**2 / 0.02))

def function_p300_nontarget(times, peak=0.3, amplitude=15.0):
    """Function to generate a p300 nontarget waveform"""
    return (1e-9 * (amplitude/4.0) * np.cos(17.*(times-peak)) *
            np.exp(-(times - peak+0.04) ** 2 / 0.05) +
            1e-9 * (amplitude/6.0) * np.sin(17.*(times-peak))*
            np.exp(-(times - peak+0.04+0.2) ** 2 / 0.02))
