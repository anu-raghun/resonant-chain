import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import cProfile
import re
cProfile.run('re.compile("foo|bar")')


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


def main():
    
    np.random.seed(0)
    npoints=100000;
    mu=1.0; timeEnd=10;
    sigma= 3*(10**(-4))
    
    (t,brightness,randNoise)=makeNullData(mu,sigma,npoints,timeEnd)
    sigmas=sigma*np.ones(len(brightness))
    tstartIndex=4
    delta=np.linspace(-0.1,0.1,1000)

    brightness=brightness+randNoise
#    diffLLTest=computeDeltaLLPerDepth(intercepts,t,mu,brightness,sigmas,tstartIndex,widthPoints)

    widthPoints=[30]
    tStartPoints=range(1,10000)
    
    f1=plt.figure(1)
    logPrediction=np.zeros(len(widthPoints))
    rootVar=np.mean(sigmas)
    

#'''a million independent light curves  
#taking one light curve and trying a million different boxes: 
#error function of the Gaussian right now: if you're looking for periodic signals, how much does it help you to know the period in advance? 
#the statistics of box least squares if you're: 1/1000 - will be 3 sigma away from zero '''
#    
#'''never use loops--> map operator ,maps & reduces'''
#''''1 through 5 sigma:  1/1000, 32% '''
#'''make a light curve with 100,000 points, width of 30 data points, order of how many over t
#even the positive direction 
    
#do a little work on removing for loops from numpy expressions
#pickle format for data: look up
#if you write out a pickle file out of the loop
#plotting operations outside 
#c profile: object construction? 
#make 100,000 point data set, going up by factors of \sqrt{10}'''
    i=0
    j=0
    while i in range(len(widthPoints)):
        currentWidth=widthPoints[i]
        logPrediction[i]=np.log(5*rootVar)-(0.5*np.log(currentWidth))
        while j in range(len(tStartPoints)):
            tCurrent=tStartPoints[j]
            diffLL=computeDeltaLLPerDepth(delta,t,mu,brightness,sigmas,tCurrent,currentWidth)
            (a,b,c)=calc_parabola_vertex(delta[5], diffLL[5], delta[6], diffLL[6], delta[4], diffLL[4])
            target=12.5
            intercepts=parabolic_roots(a,b,c,target)
            
            logDepth=np.log(intercepts[0])
            logN=np.log(currentWidth)
            plt.plot(logN,logDepth,'.k')
        i=i+1
        j=j+1
            
    
    
    logNs=np.log(widthPoints)
    plt.plot(logNs,logPrediction,'b')
    plt.title('ln(depth) vs. ln(box width)')
    plt.ylabel('ln(depth)')
    plt.xlabel('ln(box width)')
    plt.show()
    
#    plt.rcParams['agg.path.chunksize'] = 10000
#    y = (1/2)*(1+erf((brightness-mu)/(sigma*np.sqrt(2))));
#
#
#    f3=plt.figure(2)
#    plt.plot(brightness,y)
#    plt.xlabel('Brightness')
#    plt.ylabel('CDF')

#    
#    f1=plt.figure(1)
#    plt.title('Brightness v. Time')
#    plt.ylabel('brightness')
#    plt.xlabel('time')
#    plt.errorbar(t,brightness, yerr=sigmas, xerr=None,fmt='None')
#    plt.plot(t,brightness,'.k')
#    minIndex=np.argmax(brightness)
#    maxIndex=np.argmin(brightness)    
#
#
#    

#    
#    
#    
#    newDelta=np.linspace(-0.1,0.1,100000)
#    diffLLY=np.ones(len(newDelta))
#    for i in range(len(newDelta)):
#        foo=newDelta[i]
#        diffLLY[i]=a*(foo**2)+b*foo+c 
#    
#    
#    f1=plt.figure(1)
#    axes = plt.gca()
#    axes.set_ylim([-1.0,16.0])
#    plt.title('Delta Ln Likelihood vs. Depth of Box')
#    plt.ylabel('diff. ln likelihood')
#    plt.xlabel('box depth')
#    plt.plot(delta,diffLL,'b.',newDelta,diffLLY,'c-',delta,target*np.ones(len(delta)),'r')
#
#    plt.plot(intercepts[0],diffLLTest[0],'k.',intercepts[1],diffLLTest[1],'k.')
    

#    f2=plt.figure(2)
#    rootVar=np.mean(sigmas)
#    prediction=(5*rootVar)/(np.sqrt(widthPoints))
#    print(intercepts,prediction)
#
#    boxLine=boxModel(t,mu,t[tstartIndex],t[tstartIndex+widthPoints],intercepts[0])
#    boxLine2=boxModel(t,mu,t[tstartIndex],t[tstartIndex+widthPoints],intercepts[1])
#    plt.title('Brightness v. Time')
#    plt.ylabel('brightness')
#    plt.xlabel('time')
#    plt.errorbar(t,brightness, yerr=sigmas, xerr=None,fmt='none')
#    plt.plot(t,brightness,'k.',t,boxLine,'r',t,boxLine2,'r')
#    plt.show()
#    
#
#    plt.show()

    

main()


#
#