import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.special import erf
import cProfile
import random
import re
#cProfile.run('re.compile("foo|bar")')

def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
    denom = (x1-x2) * (x1-x3) * (x2-x3)
    A     = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
    B     = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
    C     = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom;
    return A,B,C
        
def makeNullData(mu,sd,nPoints,timeEnd):
    t = np.arange(nPoints)
    brightness = mu * np.ones(nPoints)
    randNoise = np.random.normal(0,sd,nPoints)
    return(t,brightness,randNoise)

def makeOneTransitData(mu,sd,nPoints,timeEnd):
    t = np.arange(nPoints)
    brightness = mu * np.ones(nPoints)
    randomIndex=random.randint(0,nPoints)
    brightness[randomIndex]=brightness[randomIndex]-1
    randNoise = np.random.normal(0,sd,nPoints)
    return(t,brightness,randNoise)

def computeDeltaLLPerDepth(delta,t,mu,brightness,noise,startIndex,period):
    diffLL=np.zeros(len(delta))
    for i in range(len(delta)):
        ll=0
        for j in range(startIndex,startIndex+period):
            temp=(brightness[j]-mu+delta[i])**2-(brightness[j]-mu)**2
            temp=temp/(2*noise[j]**2)
            ll+=temp
        diffLL[i]=ll
    return diffLL

def computeDeltaLL(delta,t,mu,brightness,noise,startIndex,period):
    diffLL=0
    for j in range(startIndex,startIndex+period):
        temp=(brightness[j]-mu+delta)**2-(brightness[j]-mu)**2
        temp=temp/(2*noise[j]**2)
        diffLL+=temp
    return diffLL
    

def boxModel(t,mu,tstart,tend,depth):
    line=np.zeros(len(t))
    for i in range(len(t)):
        if t[i]<tstart:
            line[i]=mu
        elif t[i]<= tend and t[i]>=tstart:
            muNew=(mu-depth)
            line[i]=muNew
        elif (t[i]>tend):
            line[i]=mu
    return line

def nullModel(t,mu):
    line=mu*np.ones(len(t))
    return line

def parabolic_roots(a,b,c,target):
    foo=c-target
    quad=(b*b) - (4*a*foo)
    if quad<0:
        return np.nan, np.nan
    quad=np.sqrt(quad)
    return (-b+quad)/(2*a), (-b-quad)/(2*a)

def runningDiffLnLikeFixedDepth(tStartPoints,delta,t,mu,brightness,sigmas,widthPoints):
    fixedWidthFixedDeltaLL=np.ones(len(tStartPoints))
    i=0
    while i<len(tStartPoints):
        fixedWidthFixedDeltaLL[i]=computeDeltaLL(delta,t,mu,brightness,sigmas,tStartPoints[i],widthPoints)
        i=i+1
    return fixedWidthFixedDeltaLL
    

def nullMain():
    np.random.seed(0)
    npoints=100000;
    mu=1.0; timeEnd=10;
    sigma= 3*(10**(-4))
    
    (t,brightness,randNoise)=makeNullData(mu,sigma,npoints,timeEnd)
    sigmas=sigma*np.ones(len(brightness))
#    tstartIndex=4
#    delta=np.linspace(-0.1,0.1,1000)

    brightness=brightness+randNoise
#    diffLLTest=computeDeltaLLPerDepth(intercepts,t,mu,brightness,sigmas,tstartIndex,widthPoints)

    widthPoints=30
    tStartPoints=range(1,10000)

#    rootVar=np.mean(sigmas)
    diffLL1=runningDiffLnLikeFixedDepth(tStartPoints,0.01,t,mu,brightness,sigmas,widthPoints)
    diffLL2=runningDiffLnLikeFixedDepth(tStartPoints,0.05,t,mu,brightness,sigmas,widthPoints)
    diffLL3=runningDiffLnLikeFixedDepth(tStartPoints,0.1,t,mu,brightness,sigmas,widthPoints)

    num_bins = 100
    # the histogram of the data
    
    fig, ax = plt.subplots()
    plt.hist(diffLL3, num_bins, density=True, facecolor='blue', alpha=0.3)
#    plt.hist(diffLL2, num_bins, density=True, facecolor='orange', alpha=0.3, label='$\Delta=0.05$')
#    plt.hist(diffLL1, num_bins, density=True, facecolor='green', alpha=0.3, label='$\Delta=0.1$')

    plt.xlabel('Delta Ln Like')
    plt.title(r'Histogram of Delta Ln Like: $\tau=30$, $\Delta=0.1$')
    
    #prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.show()
    

def oneTransitMain():
    np.random.seed(0)
    npoints=100000;
    mu=1.0; timeEnd=10;
    sigma= 3*(10**(-4))
    
    (t,brightness,randNoise)=makeOneTransitData(mu,sigma,npoints,timeEnd)
    sigmas=sigma*np.ones(len(brightness))
    brightness=brightness+randNoise
    
    widthPoints=30
    tStartPoints=range(1,10000)

    diffLL1=runningDiffLnLikeFixedDepth(tStartPoints,0.01,t,mu,brightness,sigmas,widthPoints)
    diffLL2=runningDiffLnLikeFixedDepth(tStartPoints,0.05,t,mu,brightness,sigmas,widthPoints)
    diffLL3=runningDiffLnLikeFixedDepth(tStartPoints,0.1,t,mu,brightness,sigmas,widthPoints)

    num_bins = 100
    # the histogram of the data
    
    fig, ax = plt.subplots()
    plt.hist(diffLL3, num_bins, density=True, facecolor='blue', alpha=0.3)
#    plt.hist(diffLL2, num_bins, density=True, facecolor='orange', alpha=0.3, label='$\Delta=0.05$')
#    plt.hist(diffLL3, num_bins, density=True, facecolor='green', alpha=0.3, label='$\Delta=0.1$')

    plt.xlabel('Delta Ln Like (data w/ one transit)')
    plt.title(r'Histogram of Delta Ln Like: $\tau=30$, $\Delta=0.1$')
    
    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.show()
    
 
    
nullMain()
#oneTransitMain()