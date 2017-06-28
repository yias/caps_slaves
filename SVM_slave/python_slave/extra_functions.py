# -*- coding: utf-8 -*-
"""

@author: jason
"""

import numpy as np
from scipy import signal
import math
import operator

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


def OnlinePreprocGonio(signal,SR,B_elbowJoint,A_elbowJoint,B_elbowVel,A_elbowVel,twLength):
    
    # filter singal
    
    FTraj=signal.lfilter(B_elbowJoint,A_elbowJoint,signal)
    
    # take the derivative of the signal
    
    dTraj=np.diff(FTraj)/(1/SR);
    
    # filtering velocity
    
    AVel=signal.lfilter(B_elbowVel,A_elbowVel,dTraj)
    
    # extract the data that correspond to the time window
    
    FTraj=FTraj[np.floor(twLength*SR):]
    
    mean_Vel=np.mean(AVel[np.floor(twLength*SR):],dtype=np.float32)
    
    return mean_Vel, FTraj




def waveformlength(sequence):
    
    sum_length=0
    
    for i in range(1,np.size(sequence)):
        Sub=np.absolute(sequence[i]-sequence[i-1])
        sum_length=sum_length+Sub
    
    return sum_length

def slopChanges(sequence,e):
    
    count_slops=0
    
    for i in xrange(0,np.shape(sequence)-2*e+1,2*e):
        SubA=sequence[i+e]-sequence[i]
        SubB=sequence[i+e]-sequence[i+2*e]
        if SubA>0 & SubB>0:
            count_slops=count_slops+1
    
    return count_slops

def majorityVote(Arr,nbClases):
    
    Uclasses=np.unique(Arr)    
    
    arr=Arr.tolist()
    
    countersAp=np.zeros([nbClases],dtype=int)    
    
    for i in range(1,nbClases+1):
        countersAp[i-1]=arr.count(i)
    
    index, value = max(enumerate(arr), key=operator.itemgetter(1))
    
    ab=np.sort(countersAp)
    
    conf=ab[-1]-ab[int(len(countersAp))-2]/int(len(arr))
    
    return index+1,conf

    
    
    