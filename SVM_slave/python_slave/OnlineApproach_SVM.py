# -*- coding: utf-8 -*-
"""

@author: jason
"""

import numpy as np
from scipy import signal
import scipy.io as sio
import extra_functions

import sys
sys.path.append('./media/jason/data/MATLAB/CBM/libsvm-3.18/python')

import svmutil

import math
import mtrx
import pce

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
    
# transfer function coefficients for the filters
B_H=0
A_H=0
B_L1=0
A_L1=0
B_L2=0
A_L2=0
B_elbowJoint=0
A_elbowJoint=0
B_elbowVel=0
A_elbowVel=0

"""
    
parameters and variable for the control scheme
    
"""
    
# the thesholds for the confidence of the majority vote and the minimun number of the time windows
    
MV_Conf_Threshold=0.7
Least_TW=12
    
# counter of the time windows\
    
counter=1
    
# a vector to contain the classification outcome of all the time windows
    
allTWOutputs=np.array([],dtype=int);
    
# number of channels
    
nb_channels=12
    
# history of the goniometer to keep
    
gonioHistory=np.zeros([np.floor(SR*twLength),1],dtype=np.float64)
    
# history of the emg signals to keep
    
emgHistory=np.zeros([np.floor(SR*twLength),nb_channels],dtype=np.float64)
    
# time windows to take into account
    
TWHistory=8

SVMmodelallmotion=None
SVMmodelonlylast=None
maxValues=None
mvc=None


def init():
    
    global B_H, A_H, B_L1, A_L1, B_L2, A_L2, B_elbowJoint, A_elbowJoint, B_elbowVel, A_elbowVel
    global SVMmodelallmotion, SVMmodelonlylast, maxValues, mvc
    
    
    """
    
    filter parameters for the EMG
    
    """
    
    # cut-off frequencies (Hz) of the band-pass filter for the EMG signals
    
    bandPassCuttOffFreq=np.array([50.0,400.0])
    
    # cut-off frequency (Hz) of the low-pass filter for the EMG signals
    
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
    
    # cut-off frequency (Hz) of the low-pass filter for the elbow joint angle
    
    cutoff_LP=50.0
    
    # order of the low-pass filter for the elbow joint angle
    
    order_LP=2;
    
    # cut-off frequency (Hz) of the low-pass filter for the angular velocity ofthe elbow joint 
    
    velLPCutOffFreq=5; 
    
    # compute the transfer coefficients for the elbow joibt angle
    
    Wn=(cutoff_LP*2)/SR;
    
    B_elbowJoint,A_elbowJoint = signal.butter(order_LP, Wn, 'low')
    
    # compute the transfer coefficients for the elbow joibt angle
    
    Wn=(velLPCutOffFreq*2)/SR;
    
    B_elbowVel,A_elbowVel = signal.butter(order_LP, Wn, 'low')

    """
    
    load classification models, mcv and max values
    
    """
    
    SVMmodelallmotion = svmutil.svm_load_model('SVMmodelallmotion.model')
    
    SVMmodelonlylast= svmutil.svm_load_model('SVMmodelonlylast.model')
    
    mat_contents = sio.loadmat('maxValues.mat')
    
    maxValues=mat_contents['maxValues'][0]
    
    mat_contents = sio.loadmat('mvc.mat')
    
    mvc=mat_contents['mvc'][0]



def run():
    
    
    global B_H, A_H, B_L1, A_L1, B_L2, A_L2, B_elbowJoint, A_elbowJoint, B_elbowVel, A_elbowVel
    global SVMmodelallmotion, SVMmodelonlylast, maxValues, mvc    
    global MV_Conf_Threshold, Least_TW, counter, allTWOutputs, nb_channels, gonioHistory, emgHistory, TWHistory
    
    """
    
    online control scheme
    
    """
    
    # acquire data
    
    np_daq_data=pce.get_var('DAQ_DATA').to_np_array()
    
    dd=np_daq_data.transpose()/float(math.pow(2,16)-1)*10-5
    
    # filter emg signals
    
    emgSignals=extra_functions.OnlinePreprocEMG(np.concatenate((emgHistory,dd[:,0:nb_channels-1]),axis=0),SR,B_H,A_H,B_L1,A_L1,B_L2,A_L2,normalize,rectify,mvc,twLength)
    
    emgHistory=dd[:,0:nb_channels-1]
    
    # feature extraction
    
    twFeatures=np.array([np.sqrt(np.mean(emgSignals[:,0]**2)),extra_functions.waveformlength(emgSignals[:,0]),extra_functions.slopChanges(emgSignals[:,0],3)])
    
    for i in range(1,nb_channels-1):
        twFeatures=np.concatenate((twFeatures,np.array([np.sqrt(np.mean(emgSignals[:,i]**2)),extra_functions.waveformlength(emgSignals[:,i]),extra_functions.slopChanges(emgSignals[:,i],3)])),axis=1)
    
    twFeatures=twFeatures/maxValues
    
    # filter goniometer data
    
    angVel,filtGonio = extra_functions.OnlinePreprocGonio(np.concatenate((gonioHistory,dd[:,nb_channels]),axis=0),SR,B_elbowJoint,A_elbowJoint,B_elbowVel,A_elbowVel,twLength)
    
    gonioHistory=dd[:,nb_channels]
    
    
    # classify emg signals
    
    if typController==1:
        timeWindowOutput, p_acc, p_val = svmutil.svm_predict([1], twFeatures, SVMmodelallmotion)
    else:
        timeWindowOutput, p_acc, p_val = svmutil.svm_predict([1], twFeatures, SVMmodelonlylast)
    
    
    allTWOutputs=np.concatenate((allTWOutputs,timeWindowOutput),axis=0)
    
    if counter>Least_TW:
        winner,conf= extra_functions.majorityVote(allTWOutputs[np.size-TWHistory:])
        
        if conf>=MV_Conf_Threshold:
            tmpClass=np.zeros([1,8],dtype=int)
            tmpClass[0]=cclasses[winner-1]
            CLAS_OUT=tmpClass
            pce.set_var('CLAS_OUT',CLAS_OUT)

    counter=counter+1;



def dispose():
    pass

