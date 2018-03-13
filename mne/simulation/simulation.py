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

from mne.simulation import simulate_sparse_stc
from mne.forward import apply_forward_raw
from mne.utils import warn,logger
from mne.io import RawArray
from mne import create_info

import numpy as np

from functions import _get_function
#Use the class simulation to specify the kind of simulation necessary

class Simulation(dict):
    """ Simulation of meg/eeg data

    Parameters
    ----------
    fwd : Forward
        a forward solution containing an instance of Info and src
    n_dipoles : int
        Number of dipoles to simulate.
    window_times : array | list | str
        time window(s) to generate activity
        If list, its size should be len(n_dipoles)
        If str, should be 'all' (default)
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
    function: list of callable | string
        To simulate a function (activity) on each dipole.
        Its size must be n_dipoles or 1 to generate the same activity on each
        dipole.
        """


    def __init__(self, fwd, n_dipoles=2, window_times='all',
                 labels=None, location='random', subject = None,
                 subjects_dir=None, function='sin'):
        self.fwd = fwd # TODO: check fwd
        
        if labels is not None:
            labels = self._check_labels(labels, n_dipoles)
        self.update(n_dipoles=n_dipoles, labels=labels, subject=subject, 
                    subjects_dir=subjects_dir, location=location)

        self.functions = self._check_function(function)
        self.window_times = self._check_window_times(window_times)


    def _check_labels(self,labels, n_dipoles):
        if n_dipoles != len(labels):
            warn('The number of labels is different from the number of '
                 'dipoles. %s dipole(s) will be generated.'
                 % min(n_dipoles, len(labels)))
        labels = labels[:n_dipoles] if n_dipoles < len(labels) else labels
        return labels


    def _check_function(self,function):
        """ Check the function given as imput with respect to number of
            dipoles. Return a list of callables.
        """
        if isinstance(function,str):
            return [_get_function(function)]
        elif isinstance(function,list):
            if len(function) != self['n_dipoles']:
                self['n_dipoles'] = min(self['n_dipoles'], len(function))
                warn('The number of functions is different from the number of '
                     'dipoles. %s function(s) will be generated.'
                     % self['n_dipoles'])
            function = function[:self['n_dipoles']]
            return [_get_function(f) for f in function]
        else:
            warn('Urecognised type. Sinusoide will be generated.')
            return [_get_function('sin')]


    def _check_window_times(self,window_times):
        """ Check the window times given as imput with respect to number of
            dipoles. Return a list of window_times.
        """

        if isinstance(window_times,list):
            if len(window_times)!=len(self.functions) and len(window_times)!=1:
                n_func = min(len(self.functions), len(window_times))
                warn('The number of window times is different from the number '
                     'of function. %s function(s) will be generated.'
                     % n_func)
            window_times = window_times[:n_func]
        else:
            window_times = [window_times]

        def get_window_time(w):
            if isinstance(w,np.ndarray) or isinstance(w,list):
                return w
            elif w!='all':
                warn('Urecognised type. '
                     'Will generated signal over whole time.')
            return 'all'

        return [get_window_time(w) for w in window_times]


    def _check_events(self,events, times):
        """ Check the window times given as imput with respect to number of
            dipoles. Return a list of window_times.
        """

        if isinstance(events,list):
            if len(events) != self.functions and events is not None:
                n_func = min(len(self.functions), len(events))
                warn('The number of event arrays is different from the number '
                     'of function. %s event arrays(s) will be generated.'
                     % n_func)
            events = events[:n_func]
        else:
            events = [events]

        def get_event(e):
            if isinstance(e,np.ndarray) or isinstance(e,list):
                if len(e)>len(times):
                    warn('The size of the event array is not the same as the '
                         'time points in the simulations. The final length of '
                         'the array will be %s.' % len(times))
                    e = e[:len(times)]
                e = np.append(np.array(e), np.zeros((len(times)-len(e))))
                return e
            elif e is not None:
                warn('Urecognized type. '
                     'Will generated signal whithout events.')
            return None

        return [get_event(e) for e in events]


    def simulate_raw_signal(self, times, cov, events=None, verbose=None,
                            random=None):
        """ Simulate a raw signal

        Parameters
        ----------
        times : array
            Time array
        cov : Covariance
            Covariance of the noise
        events : array | list | None
            events corresponding to some stimulation.
            If list, its size should be len(n_dipoles)
            If None, defaults to no event (default)
        verbose : bool, str, int, or None
            If not None, override default verbose level (see :func:`mne.verbose`
            and :ref:`Logging documentation <tut_logging>` for more).

        Returns
        -------
        raw : instance of Raw
            The simulated raw file.
        """


        if len(times) <= 2:  # to ensure event encoding works
            raise ValueError('stc must have at least three time points')

        info = self.fwd['info'].copy()
        info['sfreq'] = np.floor(len(times)/times[-1])
        info['projs'] = []
        info['lowpass'] = None
        
        #TODO: check times and time windows


        #TODO: generate data for blinks and such things - a bit hard in the
        # way it is done in simulate_raw, because the blinks and ecg are
        # computed on a dipole and then a special fwd solution is created.

        raw_data = np.zeros((len(info['ch_names']), len(times)))

        events = self._check_events(events,times)
        
        logger.info('Simulating signal from %s sources' % self['n_dipoles'])
        
        for dipoles, labels, window_time, event, data_fun in \
            self._iterate_simulation_sources(events, times):

            source_data = simulate_sparse_stc(self.fwd['src'], dipoles, 
                                              window_time, data_fun, labels, 
                                              None, self['location'],
                                              self['subject'], 
                                              self['subjects_dir'])

            if event is not None:
                from scipy.linalg import toeplitz,pinv
                #Create toeplitz array
                index = events != 0
                trig = np.zeros((len(event)))
                trig[index] = 1
                tpl = toeplitz(trig[0:len(window_time)], trig)
                propagation = np.dot(pinv(np.dot(tpl, tpl.T)), tpl)
                source_data = np.dot(source_data,propagation)

                #TODO: what if event is none?

            raw_data+=apply_forward_raw(self.fwd, source_data, info,
                                        verbose=verbose).get_data()

            #TODO: add noise using cov

        
        
        
        raw = RawArray(raw_data, info, verbose=verbose)
        logger.info('Done')
        return raw


    def _iterate_simulation_sources(self,events,times):
        """ Docstring """

        def get_window_times(w):
            if w=='all':
                return times
            else:
                return w[:len(times)]

        if len(self.functions)==1:
            yield (self['n_dipoles'], self['labels'],
                   get_window_times(self.window_times[0]),
                   events[0], self.functions[0])
        else:
            dipoles = 1
            for l,data_fun in enumerate(self.functions):
                n_wt = min(l,len(self.window_times)-1)
                n_ev = min(l,len(events)-1)
                labels = None
                if self['labels'] is not None:
                    labels = [self['labels'][l]] 
                yield (dipoles, labels,
                       get_window_times(self.window_times[n_wt]),
                       events[n_ev], data_fun)






#    simulate_sparse_stc(src, n_dipoles, times,
#                        data_fun=lambda t: 1e-7 * np.sin(20 * np.pi * t),
#                        labels=None, random_state=None, location='random',
#                        subject=None, subjects_dir=None, surf='sphere')
#
#    apply_forward_raw(fwd, stc, info, start=None, stop=None,
#                      verbose=None)