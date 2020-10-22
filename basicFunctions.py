import numpy as np
import matplotlib.pyplot as plt
#from scipy import stats

#using periodic data with HALF DURATION ITERANTS

def get_insides(ts,period,offset,halfduration):
    insides=np.abs(np.mod(ts-offset,period))<halfduration
    outsides=np.logical_not(insides)
    return(insides,outsides)
    
def periodicBoxModel(t,mu,startPointMid,offsetPeriod,halfDuration,depth):
    line=np.zeros(len(t))+mu
    offsetIndex=startPointMid
    while (offsetIndex+halfDuration)<len(t):
        line[np.arange(offsetIndex=halfDuration,offsetIndex+halfDuration)]-=depth
        offsetIndex+=offsetPeriod
    return line

def makePeriodicData(mu,sd,nPoints,period,offset,offsetDuration):
    t = np.arange(nPoints)
    brightness = mu * np.ones(nPoints)
    randNoise = np.random.normal(0,sd,nPoints)
    offsetStarts=np.arange(0, nPoints, period)
    brightness=brightness+randNoise
    for i in offsetStarts:
        brightness[i:i+offsetDuration]=brightness[i:i+offsetDuration]-offset
    return(t,brightness)
