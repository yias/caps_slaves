# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 10:53:02 2017

@author: jason
"""

import numpy as np
from scipy import signal

# which model to use:
# 1 -> use data including the reaching motion
# 2 -> use data without including the reaching motion

typController=1
 
 # classes correspondence
 
cclasses=np.array([16,16,18,21])

# sample rate

SR=1000.0;

# time window length in seconds

twLength=0.150

# normalization of the EMG signals by the MVC

normalize=True;

# rectifications of the EMG signals

rectify=True

"""

filter parameters for the EMG

"""

# cut-off frequencies of the band-pass filter for the EMG signals

bandPassCuttOffFreq=np.array([50.0,400.0])

# cut-off frequency of the low-pass filter for the EMG signals

lowPassCutOffFreq=20.0

# order of the filter

filter_order=7;

# compute the transfer function coefficients for the EMG filtering

Wn=(bandPassCuttOffFreq[0]*2)/SR

B_H,A_H = signal.butter(filter_order,Wn,'high')

Wn=(bandPassCuttOffFreq[1]*2)/SR

B_L1,A_L1 = signal.butter(filter_order,Wn,'low')

Wn=(lowPassCutOffFreq*2)/SR

B_L2,A_L2 = signal.butter(filter_order,Wn,'low')

"""

filter parameters for the Goniometer data

"""

# cut-off frequency of the low-pass filter for the elbow joint angle

cutoff_LP=50.0

# order of the low-pass filter for the elbow joint angle

print("{:.6f}".format(Wn))