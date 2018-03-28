# author: ngayraud.
#
# Created on Tue Feb 20 11:14:54 2018.

import numpy as np

from .source import simulate_sparse_stc
from ..forward import apply_forward_raw
from ..utils import warn, logger
from ..io import RawArray
from ..io.constants import FIFF

from .functions import get_function
from .noise import generate_noise_data


class Simulation(dict):
    """Simulation of meg/eeg data.

    Parameters
    ----------
    fwd : Forward
        a forward solution containing an instance of Info and src
    n_dipoles : int
        Number of dipoles to simulate.
    labels : None | list of Labels
        The labels. The default is None, otherwise its size must be n_dipoles.
    location : str
        The label location to choose a dipole from. Can be ``random`` (default)
        or ``center`` to use :func:`mne.Label.center_of_mass`. Note that for
        ``center`` mode the label values are used as weights.
    subject : string | None
        The subject the label is defined for.
        Only used with location=``center``.
    subjects_dir : str, or None
        Path to the SUBJECTS_DIR. If None, the path is obtained by using the
        environment variable SUBJECTS_DIR. Only used with location=``center``.
    function: list of callables/str of length n_dipoles | str | callable
        To simulate a function (activity) on each dipole. If it is a string or
        a callable, the same activity will be generated over all dipoles
    window_times : array | list | str
        time window(s) to generate activity. If list, its size should be 
        len(function). If str, should be ``all`` (default)

    Notes
    -----
    Some notes.
    """

    def __init__(self, fwd, n_dipoles=2, labels=None, location='random',
                 subject=None, subjects_dir=None, function='sin',
                 window_times='all'):
        self.fwd = fwd  # TODO: check fwd
        if labels is not None:
            labels, n_dipoles = self._check_labels(labels, n_dipoles)
        self.update(n_dipoles=n_dipoles, labels=labels, subject=subject,
                    subjects_dir=subjects_dir, location=location,
                    info=self.fwd['info'])

        self.functions = self._check_function(function)
        self.window_times = self._check_window_times(window_times)
        self['info']['projs'] = []
        self['info']['bads'] = []

    def _check_labels(self, labels, n_dipoles):
        """Check the function given as imput wrt the number of dipoles.

        Return a list of labels and the number of dipoles.
        """
        n_labels = min(n_dipoles, len(labels))
        if n_dipoles != len(labels):
            warn('The number of labels is different from the number of '
                 'dipoles. %s dipole(s) will be generated.'
                 % n_labels)
        labels = labels[:n_labels]
        return labels, n_labels

    def _check_function(self, function):
        """Check the function given as imput wrt the number of dipoles.

        Return a list of callables.
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
                pad = self['n_dipoles'] - len(function)
                warn('The number of functions is smaller from the number of '
                     'dipoles. %s sinusoid(s) will be added.'
                     % pad)
                function = function + ['sin'] * pad

            return [get_function(f) for f in function]

        else:
            warn('Urecognised type. Sinusoide will be generated.')
            return [get_function('sin')]

    def _check_window_times(self, window_times):
        """Check the window times given as input wrt the number of dipoles.

        Return a list of window_times.
        """
        if isinstance(window_times, list):

            if len(window_times) > len(self.functions):
                n_func = len(self.functions)
                warn('The number of window times is greater than the number '
                     'of functions. %s function(s) will be generated.'
                     % n_func)
                window_times = window_times[:n_func]

            elif len(window_times) < len(self.functions):
                pad = len(self.functions) - len(window_times)
                warn('The number of window times is smaller than the number '
                     'of functions. Assuming that the last ones are \'all\'')
                window_times = window_times + ['all'] * pad
        else:
            window_times = [window_times]

        def get_window_time(w_t):
            """Check if window time has the correct value."""
            if isinstance(w_t, np.ndarray):
                return w_t
            elif not isinstance(w_t, str) or w_t != 'all':
                warn('Urecognised type. '
                     'Will generated signal over whole time.')
            return 'all'

        return [get_window_time(w_t) for w_t in window_times]


def _iterate_simulation_sources(sim, events, times):
    """Iterate over all stimulation functions."""
    def correct_window_times(w_t, e_t):
        """Check if window time has the correct length."""
        if (isinstance(w_t, str) and w_t == 'all') or e_t is None:
            return times
        else:
            if len(w_t) > len(times):
                warn('Window is too large, will be cut to match the '
                     'length of parameter \'times\'')
            return w_t[:len(times)]

    if len(sim.functions) == 1:
        yield (sim['n_dipoles'], sim['labels'],
               correct_window_times(sim.window_times[0]),
               events[0], sim.functions[0])
    else:
        dipoles = 1
        for index, data_fun in enumerate(sim.functions):
            n_wt = min(index, len(sim.window_times) - 1)
            n_ev = min(index, len(events) - 1)
            labels = None
            if sim['labels'] is not None:
                labels = [sim['labels'][index]]
            yield (dipoles, labels,
                   correct_window_times(sim.window_times[n_wt], events[n_ev]),
                   events[n_ev], data_fun)


def _check_event(event, times):
    """Check if event array has the correct shape/length."""
    if isinstance(event, np.ndarray) and event.shape[1] == 3:
        if np.max(event) > len(times) - 1:
            warn('The indices in the event array is not the same as '
                 'the time points in the simulations.')
            event[np.where(event > len(times) - 1), 0] = len(times) - 1
        return np.array(event)
    elif event is not None:
        warn('Urecognized type. Will generated signal without events.')
    return None


def get_events(sim, times, events):
    """Get a list of events.

    Checks if the input events correspond to the simulation times.

    Parameters
    ----------
    sim : instance of Simulation
        Initialized Simulation object with parameters
    times : array
        Time array
    events : array,  | list of arrays | None
        events corresponding to some stimulation.
        If array, its size should be shape=(len(times), 3)
        If list, its size should be len(n_dipoles)
        If None, defaults to no event (default)

    Returns
    -------
    events : list
        a list of events of type array, shape=(n_events, 3) | None
    """
    if isinstance(events, list):
        if len(events) > sim.functions:
            n_func = len(sim.functions)
            warn('The number of event arrays is greater than the number '
                 'of functions. %s event arrays(s) will be generated.'
                 % n_func)
            events = events[:n_func]
        elif len(events) < sim.functions:
            pad = len(sim.functions) - len(events)
            warn('The number of event arrays is smaller than the number '
                 'of functions. Assuming that the last ones are None')
            events = events + [None] * pad
    else:
        events = [events]

    return [_check_event(event, times) for event in events]


def simulate_raw_signal(sim, times, cov=None, events=None, random_state=None,
                        verbose=None):
    """Simulate a raw signal.

    Parameters
    ----------
    sim : instance of Simulation
        Initialized Simulation object with parameters
    times : array
        Time array
    cov : Covariance | string | dict | None
        Covariance of the noise
    events : array, shape = (n_events, 3) | list of arrays | None
        events corresponding to some stimulation.
        If list, its size should be len(n_dipoles)
        If None, defaults to no event (default)
    random_state : None | int | np.random.RandomState
        To specify the random generator state.
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

    info = sim['info'].copy()
    # TODO: check frequencies
    info['sfreq'] = np.floor(len(times) / (times[-1] - times[0]))
    info['projs'] = []
    info['lowpass'] = None

    # TODO: generate data for blinks and such things - a bit hard in the
    # way it is done in simulate_raw, because the blinks and ecg are
    # computed on a dipole and then a special fwd solution is created.

    raw_data = np.zeros((len(info['ch_names']), len(times)))

    events = get_events(sim, times, events)

    logger.info('Simulating signal from %s sources' % sim['n_dipoles'])

    for dipoles, labels, window_time, event, data_fun in \
            _iterate_simulation_sources(sim, events, times):

        source_data = simulate_sparse_stc(sim.fwd['src'], dipoles,
                                          window_time, data_fun, labels,
                                          None, sim['location'],
                                          sim['subject'], sim['subjects_dir'])

        propagation = _get_propagation(event, times, window_time)
        source_data.data = np.dot(source_data.data, propagation)
        raw_data += apply_forward_raw(sim.fwd, source_data, info,
                                      verbose=verbose).get_data()

    # Noise
    if cov is not None:
        raw_data += generate_noise_data(info, cov, len(times), random_state)[0]

    stim_chan = dict(ch_name='STI 014', coil_type=FIFF.FIFFV_COIL_NONE,
                     kind=FIFF.FIFFV_STIM_CH, logno=len(info["chs"]) + 1,
                     scanno=len(info["chs"]) + 1, cal=1., range=1.,
                     loc=np.full(12, np.nan), unit=FIFF.FIFF_UNIT_NONE,
                     unit_mul=0., coord_frame=FIFF.FIFFV_COORD_UNKNOWN)
    info['chs'].append(stim_chan)
    info._update_redundant()
    raw_data = np.vstack((raw_data, np.zeros((1, len(times)))))

    # Create RawArray object with all data
    raw = RawArray(raw_data, info, first_samp=times[0], verbose=verbose)

    # If events exist, add them
    stimulations = [event for event in events if event is not None]
    if len(stimulations) != 0:
        stimulations = np.unique(np.vstack(stimulations), axis=0)
        # Add events onto a stimulation channel
        raw.add_events(stimulations, stim_channel='STI 014')

    logger.info('Done')
    return raw


def _get_propagation(event, times, window_time):
    """Return the matrix that propagates the waveforms."""
    propagation = 1.0

    if event is not None:

        # generate events 1d array
        e_tmp = np.zeros(len(times))
        e_ind = np.array(event[:, 0], dtype=int)
        e_tmp[e_ind] = event[:, 2]

        from scipy.linalg import toeplitz
        # Create toeplitz array
        index = e_tmp != 0
        trig = np.zeros((len(e_tmp)))
        trig[index] = 1
        propagation = toeplitz(trig[0:len(window_time)], trig)

    return propagation
