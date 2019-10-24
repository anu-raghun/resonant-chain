import numpy as np
import matplotlib.pyplot as plt

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
    brightness=brightness+randNoise
    return(t,brightness)

def computeDeltaLLPerDepth(depth,t,mu,brightness,noise,startIndex,duration):
    diffLL=np.zeros(len(depth))
    for i in range(len(depth)):
        ll=0
        for j in range(startIndex,startIndex+duration):
            temp=(brightness[j]-mu+depth[i])**2-(brightness[j]-mu)**2
            temp=temp/(2*noise[j]**2)
            ll+=temp
        diffLL[i]=ll
    return diffLL

def computeDeltaLL(depth,mu,brightness,noise,startIndex,duration):
    JJ=np.arange(startIndex,startIndex+duration)
    d2=-0.5*(brightness[JJ]-(mu-depth))**2/noise[JJ]**2
    d2Null=-0.5*(brightness[JJ]-mu)**2/noise[JJ]**2
    return np.sum(d2-d2Null)

def boxModel(t,mu,tstart,tend,depth):
    line=np.zeros(len(t))+mu
    line[np.arange(tstart,tend)]-=depth
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

def runningDiffLnLikeFixedDepth(tStartPoints,depth,mu,brightness,sigmas,widthPoints):
    fixedWidthFixedDeltaLL=np.ones(len(tStartPoints))
    i=0
    while i<len(tStartPoints):
        fixedWidthFixedDeltaLL[i]=computeDeltaLL(depth,mu,brightness,sigmas,tStartPoints[i],widthPoints)
        i=i+1
    return fixedWidthFixedDeltaLL
    

def nullMain():
    np.random.seed(0)
    npoints=10;
    mu=1.0; timeEnd=10;
    sigma= 2*(10**(-12))
    
    (t,brightness)=makeNullData(mu,sigma,npoints,timeEnd)
    sigmas=sigma*np.ones(len(brightness))
    
    duration=3
    tStartPoint=1
    depth=(2**(-12))

#    diffLL1=runningDiffLnLikeFixedDepth(tStartPoints,depthCurrent,mu,brightness,sigmas,widthPoints)
    diffLL1=computeDeltaLL(depth,mu,brightness,sigmas,tStartPoint,duration)
    diffLL2=computeDeltaLL(-depth,mu,brightness,sigmas,tStartPoint,duration)
#    diffLL3=runningDiffLnLikeFixedDepth(tStartPoints,0.1,t,mu,brightness,sigmas,widthPoints)
    print(diffLL1,diffLL2)
    
# UNCOMMENT WHEN YOU WANT HISTOGRAMS
#    num_bins = 50
#    # the histogram of the data
#    
#    fig, ax = plt.subplots()
##    plt.hist(diffLL3, num_bins, density=True, facecolor='blue', alpha=0.3)
##    plt.hist(diffLL2, num_bins, density=True, facecolor='orange', alpha=0.3, label='$\depth=-0.01$')
#    plt.hist(diffLL1, num_bins, density=True, facecolor='green', alpha=0.3, label='$\depth=0.01$')
#
#    plt.xlabel('Delta Ln Like')
#    plt.title(r'Histogram of Delta Ln Like: $\tau=3$, $\depth=0.01$')
#    plt.subplots_adjust(left=0.15)
#    plt.show()
    
    f2=plt.figure(2)
    plt.plot(t,brightness)
    tstartIndex=tStartPoint
    boxLine=boxModel(t,mu,t[tstartIndex],t[tstartIndex+duration],depth)
    boxLine2=boxModel(t,mu,t[tstartIndex],t[tstartIndex+duration],-depth)
    plt.step(t,boxLine,color='r',where='mid')
    plt.step(t,boxLine2,color='r',where='mid')
    

    plt.title('Brightness v. Time')
    plt.ylabel('brightness')
    plt.xlabel('time')
    plt.errorbar(t,brightness,yerr=sigmas, xerr=None,fmt='none')
    plt.plot(t,brightness,'k.')
    plt.show()
    
    #prevent clipping of ylabel
 
    
nullMain()
#oneTransitMain()