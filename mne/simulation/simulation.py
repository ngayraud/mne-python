# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:14:54 2018

@author: ngayraud

This code assumes that we have no info on anything except for
the channels present in the forward solution (should be given through the
Info structure of mne).

The user (or programmer) provides:
 -- Number (and locations - through the Label class of mne) of desired dipoles
 -- Functions: one per dipole - to simulate some specific activity.
 -- Time duration of simulation for raw, number of epochs for epoched
 -- Time window for evoked simulations, one per dipole. If the simulation is
    a continuous signal, then no time window should be given. The code should
    take the entire simulation time.
 -- Events for evoked stimulations, one per dipole. (See previous remark)
 -- Type of noise, through a Covariance object.
"""
import numpy as np

from .source import simulate_sparse_stc
from ..forward import apply_forward_raw
from ..utils import warn, logger
from ..io import RawArray

from .functions import get_function
#Use the class simulation to specify the kind of simulation necessary

class Simulation(dict):
    """ Simulation of meg/eeg data

    Parameters
    ----------
    fwd : Forward
        a forward solution containing an instance of Info and src
    n_dipoles : int
        Number of dipoles to simulate.
    labels : None | list of Labels
        The labels. The default is None, otherwise its size must be n_dipoles.
    location : str
        The label location to choose a dipole from. Can be 'random' (default)
        or 'center' to use :func:`mne.Label.center_of_mass`. Note that for
        'center' mode the label values are used as weights.
    subject : string | None
        The subject the label is defined for. Only used with location='center'.
    subjects_dir : str, or None
        Path to the SUBJECTS_DIR. If None, the path is obtained by using the
        environment variable SUBJECTS_DIR. Only used with location='center'.
    function: list of callables/str of lenght n_dipoles | str | callable
        To simulate a function (activity) on each dipole. If it is a string or
        a callable, the same activity will be generated over all dipoles
    window_times : array | list | str
        time window(s) to generate activity
        If list, its size should be len(function)
        If str, should be 'all' (default)
        """


    def __init__(self, fwd, n_dipoles=2, labels=None, location='random',
                 subject=None, subjects_dir=None, function='sin',
                 window_times='all'):
        self.fwd = fwd # TODO: check fwd

        if labels is not None:
            labels, n_dipoles = self._check_labels(labels, n_dipoles)
        self.update(n_dipoles=n_dipoles, labels=labels, subject=subject,
                    subjects_dir=subjects_dir, location=location)

        self.functions = self._check_function(function)
        self.window_times = self._check_window_times(window_times)


    def _check_labels(self, labels, n_dipoles):
        """ Check the numver of labels given as imput with respect to number of
            dipoles. Return a list of labels and the number of dipoles.
        """
        n_labels = min(n_dipoles, len(labels))
        if n_dipoles != len(labels):
            warn('The number of labels is different from the number of '
                 'dipoles. %s dipole(s) will be generated.'
                 % n_labels)
        labels = labels[:n_labels]
        return labels, n_labels


    def _check_function(self, function):
        """ Check the function given as imput with respect to number of
            dipoles. Return a list of callables.
        """
        if isinstance(function, str):
            return [get_function(function)]

        elif isinstance(function, list):

            if len(function) > self['n_dipoles']:
                warn('The number of functions is greater from the number of '
                     'dipoles. %s function(s) will be generated.'
                     % self['n_dipoles'])
                function = function[:self['n_dipoles']]

            elif len(function) < self['n_dipoles']:
                pad = self['n_dipoles']-len(function)
                warn('The number of functions is smaller from the number of '
                     'dipoles. %s sinusoid(s) will be added.'
                     % pad)
                function = function+['sin']*pad

            return [get_function(f) for f in function]

        else:
            warn('Urecognised type. Sinusoide will be generated.')
            return [get_function('sin')]


    def _check_window_times(self, window_times):
        """ Check the window times given as imput with respect to number of
            dipoles. Return a list of window_times.
        """

        if isinstance(window_times, list):

            if len(window_times) > len(self.functions):
                n_func = len(self.functions)
                warn('The number of window times is greater than the number '
                     'of functions. %s function(s) will be generated.'
                     % n_func)
                window_times = window_times[:n_func]

            elif len(window_times) < len(self.functions):
                pad = len(self.functions)-len(window_times)
                warn('The number of window times is smaller than the number '
                     'of functions. Assuming that the last ones are \'all\'')
                window_times = window_times+['all']*pad
        else:
            window_times = [window_times]

        def get_window_time(w_t):
            """Nested function to check if window time has the correct value
            """
            if isinstance(w_t, np.ndarray):
                return w_t
            elif not isinstance(w_t, str) or w_t != 'all':
                warn('Urecognised type. '
                     'Will generated signal over whole time.')
            return 'all'

        return [get_window_time(w_t) for w_t in window_times]


    def iterate_simulation_sources(self, events, times):
        """ Iterate over the number of functions """

        def correct_window_times(w_t):
            """Nested function to check if window time has the correct length
            """
            if isinstance(w_t, str) and w_t == 'all':
                return times
            else:
                if len(w_t) > len(times):
                    warn('Window is too large, will be cut to match the '
                         'length of parameter \'times\'')
                return w_t[:len(times)]

        if len(self.functions) == 1:
            yield (self['n_dipoles'], self['labels'],
                   correct_window_times(self.window_times[0]),
                   events[0], self.functions[0])
        else:
            dipoles = 1
            for index, data_fun in enumerate(self.functions):
                n_wt = min(index, len(self.window_times)-1)
                n_ev = min(index, len(events)-1)
                labels = None
                if self['labels'] is not None:
                    labels = [self['labels'][index]]
                yield (dipoles, labels,
                       correct_window_times(self.window_times[n_wt]),
                       events[n_ev], data_fun)

    def check_events(self, events, times):
        """ Check the window times given as imput with respect to number of
            dipoles. Return a list of window_times.
        """
        if isinstance(events, list):
            if len(events) > self.functions:
                n_func = len(self.functions)
                warn('The number of event arrays is greater than the number '
                     'of function. %s event arrays(s) will be generated.'
                     % n_func)
                events = events[:n_func]
            elif len(events) < self.functions:
                pad = len(self.functions)-len(events)
                warn('The number of window times is smaller than the number '
                     'of functions. Assuming that the last ones are None')
                events = events+[None]*pad
        else:
            events = [events]

        def get_event(event):
            """ Nested function to check if event has the correct shape and
                length
            """
            if isinstance(event, np.ndarray) and event.shape[1] == 3:
                if np.max(event) > len(times)-1:
                    warn('The indices in the event array is not the same as the '
                         'time points in the simulations.')
                    event[np.where(event > len(times)-1), 0] = len(times)-1
                return event
            elif event is not None:
                warn('Urecognized type. '
                     'Will generated signal from the beginning.')
            return None

        return [get_event(event) for event in events]


def simulate_raw_signal(sim, times, cov, events=None, verbose=None):
    """ Simulate a raw signal

    Parameters
    ----------
    sim : instance of Simulation
        Initialized Simulation object with parameters
    times : array
        Time array
    cov : Covariance
        Covariance of the noise
    events : array, shape = (n_events, 3) | list of arrays | None
        events corresponding to some stimulation.
        If list, its size should be len(n_dipoles)
        If None, defaults to no event (default)
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    raw : instance of RawArray
        The simulated raw file.
    """

    if len(times) <= 2:  # to ensure event encoding works
        raise ValueError('stc must have at least three time points')

    info = sim.fwd['info'].copy()
    info['sfreq'] = np.floor(len(times)/times[-1])
    info['projs'] = []
    info['lowpass'] = None

    #TODO: generate data for blinks and such things - a bit hard in the
    # way it is done in simulate_raw, because the blinks and ecg are
    # computed on a dipole and then a special fwd solution is created.

    raw_data = np.zeros((len(info['ch_names']), len(times)))

    events = sim.check_events(events, times)

    logger.info('Simulating signal from %s sources' % sim['n_dipoles'])

    for dipoles, labels, window_time, event, data_fun in \
        sim.iterate_simulation_sources(events, times):

        source_data = simulate_sparse_stc(sim.fwd['src'], dipoles,
                                          window_time, data_fun, labels,
                                          None, sim['location'],
                                          sim['subject'], sim['subjects_dir'])

        propagation = _get_propagation(event, times, window_time)

        source_data.data = np.dot(source_data.data, propagation)

        raw_data += apply_forward_raw(sim.fwd, source_data, info,
                                      verbose=verbose).get_data()

        #TODO: add noise using cov

    raw = RawArray(raw_data, info, verbose=verbose)
    #TODO: maybe add "main" event

    logger.info('Done')
    return raw


def _get_propagation(event, times, window_time):

    propagation = 1.0

    if event is not None:

        #generate events 1d array
        e_tmp = np.zeros(len(times))
        e_ind = np.array(event[:, 0], dtype=int)
        e_tmp[e_ind] = event[:, 2]

        from scipy.linalg import toeplitz, pinv
        #Create toeplitz array
        index = e_tmp != 0
        trig = np.zeros((len(e_tmp)))
        trig[index] = 1
        tpl = toeplitz(trig[0:len(window_time)], trig)
        propagation = np.dot(pinv(np.dot(tpl, tpl.T)), tpl)

    return propagation
