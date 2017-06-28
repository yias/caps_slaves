# -*- coding: utf-8 -*-
"""

@author: jason
"""

import numpy as np
from scipy import signal
import math

def OnlinePreprocEMG(data,sr,B_H,A_H,B_L1,A_L1,B_L2,A_L2,normaLize,rectiFy,mvc,twLength):
    
    # detrend
    data_new=signal.detrend(data)
    
    # band-pass filtering
    
    #zi = signal.lfilter_zi(B_H,A_H)
    
    data_h=signal.lfilter(B_H,A_H,data_new)
    
    #zi = signal.lfilter_zi(B_L1,A_L1)
    
    data_hl=signal.lfilter(B_L1,A_L1,data_h)
    
    # rectification
    
    if rectiFy:
        data_rett=np.absolute(data_hl)
    else:
        data_rett=data_hl

    # low-pass filtering (for creating linear envelope)

    #zi = signal.lfilter_zi(B_L2,A_L2)
    EMG_pp=signal.lfilter(B_L2,A_L2,data_rett)
    
    # normalization
    
    if normaLize:
        for i in range(0,np.shape(EMG_pp)[2]):
            EMG_pp[:,i]=EMG_pp[:,i]/mvc[i]

    return EMG_pp[np.floor(twLength*sr):,:]


def 