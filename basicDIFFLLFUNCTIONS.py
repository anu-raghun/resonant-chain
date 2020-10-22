import numpy as np
import matplotlib.pyplot as plt
#from scipy import stats



def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
    denom = (x1-x2) * (x1-x3) * (x2-x3)
    A     = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
    B     = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
    C     = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom;
    return A,B,C
        
def makeNullData(mu,sd,nPoints):
    t = np.arange(nPoints)
    brightness = mu * np.ones(nPoints)
    randNoise = np.random.normal(0,sd,nPoints)
    brightness=brightness+randNoise
    return(t,brightness)
    
def makePeriodicData(mu,sd,nPoints,period,offset,offsetDuration):
    t = np.arange(nPoints)
    brightness = mu * np.ones(nPoints)
    randNoise = np.random.normal(0,sd,nPoints)
    offsetStarts=np.arange(0, nPoints, period)
    brightness=brightness+randNoise
    for i in offsetStarts:
        brightness[i:i+offsetDuration]=brightness[i:i+offsetDuration]-offset
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

def computeDeltaLLPeriodic(period,mu,startpoint,brightness,noise,duration):
    i=startpoint
    d2=d2Null=0
    while i+duration<len(brightness):
        JJ=np.arange(i,i+duration)
        depth=mu-np.mean(brightness[i:i+duration])
        d2+=np.sum(-0.5*(brightness[JJ]-(mu-depth))**2/noise[JJ]**2)
        d2Null+=np.sum(-0.5*(brightness[JJ]-mu)**2/noise[JJ]**2)
        i+=period
    return d2-d2Null

def boxModel(t,mu,tstart,tend,depth):
    line=np.zeros(len(t))+mu
    line[np.arange(tstart,tend)]-=depth
    return line

def periodicBoxModelAlterableDepth(brightness,t,mu,startpoint,offsetPeriod,boxDuration):
    line=np.zeros(len(t))+mu
    offsetIndex=startpoint
    while offsetIndex<len(t):
        depth=mu-np.mean(brightness[offsetIndex:offsetIndex+boxDuration])
        line[np.arange(offsetIndex,offsetIndex+boxDuration)]-=depth
        offsetIndex+=offsetPeriod
    return line

def periodicBoxModel(t,mu,startpoint,offsetPeriod,boxDuration,depth):
    line=np.zeros(len(t))+mu
    offsetIndex=startpoint
    while (offsetIndex+boxDuration)<len(t):
        line[np.arange(offsetIndex,offsetIndex+boxDuration)]-=depth
        offsetIndex+=offsetPeriod
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

#def makeSubplotsWithNullData(m,n,xLabel,yLabel,title,t,brightness,sigmas):
#    fig, a =  plt.subplots(2,2)
#    fig.text(0.5, 0.04, xLabel, ha='center', va='center')
#    fig.text(0.06, 0.5, yLabel, ha='center', va='center', rotation='vertical')
#    fig.suptitle(title)
#    for j in range(n):
#        for i in range(m):      
#            a[i][j].errorbar(t,brightness, yerr=sigmas,fmt='ko',markersize=0.7,ecolor='blue')
#    return fig, a
    
def fillSubplots(fig,a,m,n,t,brightness,sigmas,depth,tStartPoint,duration,mu):
    for j in range(n-1):
        for i in range(m-1):
            a[i][j].plot(boxModel(t,mu,tStartPoint,tStartPoint+duration,depth),'y')


def drawBasicBrightnessPlot(t,brightness,uncertaintyPerPoint):
    plt.errorbar(t,brightness, yerr=uncertaintyPerPoint,fmt='ko',markersize=0.7,ecolor='blue')
    plt.ylabel('brightness')
    plt.xlabel('time')
    
def drawBasicBrightnessSubplot(t,brightness,uncertaintyPerPoint,subplot):
    subplot.errorbar(t,brightness, yerr=uncertaintyPerPoint,fmt='ko',markersize=0.7,ecolor='blue')
    subplot.set(xlabel='time', ylabel='brightness')
    
def get_insides(ts,period,offset,halfduration):
    insides=np.abs(np.mod(ts-offset,period))<halfduration
    outsides=np.logical_not(insides)
    return(insides,outsides)


    