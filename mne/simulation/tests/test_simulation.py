#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 15:53:38 2018

@author: ngayraud
"""

import mne
import numpy as np
from simulation import Simulation

import warnings 

from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_equal)

from nose.tools import assert_true, assert_raises

#%%
#Parameters
def test_with_single_function(fwd_f,labels, subject, subjects_dir):
    
    freq = 256.0
    time = 2.0
    function='sin'
    
    #Simulation
    times = np.arange(0, time, 1.0/freq)
    sim_1 = Simulation(fwd_f, n_dipoles=2, labels=labels, location = 'center',
                       subject=subject, subjects_dir=subjects_dir, 
                       function=function)
    raw_1 = sim_1.simulate_raw_signal(times, cov=0, verbose=0)
    data_1,_ = raw_1[:]
    sim_2 = Simulation(fwd_f, n_dipoles=2, labels=labels, location = 'center',
                       subject=subject, subjects_dir=subjects_dir, 
                       function=function)
    raw_2 = sim_2.simulate_raw_signal(times, cov=0, verbose=0)
    data_2,_ = raw_2[:]
    
    #This should work
    assert_equal(data_1, data_2)
    assert_true(~np.isnan(np.sum(data_1)) or ~np.isinf(np.sum(data_1)))
    assert_true(~np.isnan(np.sum(data_2)) or ~np.isinf(np.sum(data_1)))
    
    #These should raise warnings
    warnings.simplefilter("error")
    
    assert_raises(RuntimeWarning,Simulation,fwd_f,
                  n_dipoles=3,labels=labels,function=function)
    assert_raises(RuntimeWarning,Simulation,fwd_f,
              n_dipoles=2,labels=labels[:1],function=function)
    assert_raises(RuntimeWarning,Simulation,fwd_f,
              n_dipoles=2,labels=labels,function='hi')
    
    warnings.simplefilter("always")
 #%%   
def test_with_multiple_functions(fwd_f,labels, subject, subjects_dir):

    freq = 256.0
    function=['p300_target','sin']
    time = 2.0
    
    #Simulation
    times = np.arange(0,time,1.0/freq)
    sim_1 = Simulation(fwd_f,n_dipoles=2, labels=labels, location = 'center',
                       subject=subject, subjects_dir=subjects_dir,
                       function=function)
    raw_1 = sim_1.simulate_raw_signal(times, cov=0, verbose=0)
    data_1,_ = raw_1[:]
    sim_2 = Simulation(fwd_f, n_dipoles=2, labels=labels, location = 'center',
                       subject=subject, subjects_dir=subjects_dir, 
                       function=function)
    raw_2 = sim_2.simulate_raw_signal(times, cov=0, verbose=0)
    data_2,_ = raw_2[:]
    
    #This should work
    assert_equal(data_1, data_2)
    assert_true(~np.isnan(np.sum(data_1)) or ~np.isinf(np.sum(data_1)))
    assert_true(~np.isnan(np.sum(data_2)) or ~np.isinf(np.sum(data_1)))
    
    #These should raise warnings    
    warnings.simplefilter("error")

    assert_raises(RuntimeWarning,Simulation,fwd_f,
                  n_dipoles=2,labels=labels,
                  function=['sin']+function)
    
    warnings.simplefilter("always")

#%%    
#Initialize
data_path = mne.datasets.sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
subjects_dir = data_path + '/subjects'
trans = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'

subject = 'sample'
#%%
#Compute forward model
src = mne.setup_source_space(subject, spacing='oct6',
                             subjects_dir=subjects_dir, add_dist=False)
conductivity = (0.3, 0.006, 0.3)
model = mne.make_bem_model(subject, ico=4,
                           conductivity=conductivity,
                           subjects_dir=subjects_dir)

bem = mne.make_bem_solution(model)
fwd = mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=bem,
                                meg=True, eeg=True, mindist=5.0, n_jobs=2)

fwd_f = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                    use_cps=True)

#%%
label_names = ['Vis-lh', 'Vis-rh']
labels = [mne.read_label(data_path + '/MEG/sample/labels/%s.label' % ln)
          for ln in label_names]
for label in labels:
    label.values[:] = 1
    
test_with_single_function(fwd_f,labels, subject, subjects_dir)
test_with_multiple_functions(fwd_f,labels, subject, subjects_dir)
#%%