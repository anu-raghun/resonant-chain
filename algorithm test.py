import basicPeriodicFunctions as bf
#import basicDIFFLLFUNCTIONS as dll
import numpy as np
import matplotlib.pyplot as plt
import periodicTestRunsMod as ptestmod
from scipy import special
from scipy.special import erfcinv
import pickle
import astropy.timeseries
import pandas as pd
import time
import seaborn as sns
import numpy.random as random
import pandas as pd 
import scipy.stats as stats





np.random.seed(0)
trange=[0,1000.7]; 
npoints=50000
mu=1.0; offsetPeriod1=18.767; offset1=13.2; transitDuration1=0.525;
randomNoiseSigma= (10**(-4))
halfDuration1=transitDuration1/2
transitDepth=0.0002
delta=0.005

def main():
    t = np.arange(trange[0],trange[1],delta)
    npoints=len(t)
    print(npoints)
    
    insides,outsides=bf.get_insides(t,offsetPeriod,offset,halfDuration)
    
    # brightnessPeriodic=bf.makePeriodicData(mu,randomNoiseSigma,len(t),offsetPeriod,offset,insides,transitDepth)
    # brightnessPeriodic_FiducialTransit=bf.makePeriodicData(mu,randomNoiseSigma,len(t),offsetPeriod,offset,insides,transitDepth+0.001)
    # brightnessPeriodic_FiducialOffset=bf.makePeriodicData(mu,randomNoiseSigma,len(t),offsetPeriod,offset+312,insides,transitDepth)
    
    'trying out per depth period'
    # depthsNULL=bf.get_depths_Per_Point(brightnessNULL,insides,mu)
    # depthsPeriodic=bf.get_depths_Per_Point(brightnessPeriodic,insides,mu)
    # depthsFiducialTransit=bf.get_depths_Per_Point(brightnessPeriodic_FiducialTransit,insides,mu)
    # depthsFiducialOffset=bf.get_depths_Per_Point(brightnessPeriodic_FiducialOffset,insides,mu)
    
    # plt.plot(t,brightnessPeriodic)
    # plt.show
    
    # plt.figure()
    # plt.hist(brightnessPeriodic,bins='auto')
    # plt.title('brightness')
    # plt.show
    
    # plt.figure()
    
    # plt.hist(depthsPeriodic,bins='auto',label='per point depth of periodic data')
    # plt.hist(depthsNULL,bins='auto',alpha=0.25,label='per point depth of null data')
    # plt.hist(depthsFiducialTransit,bins='auto',alpha=0.75,label='transit depth off by 0.001')
    # plt.hist(depthsFiducialOffset,bins='auto',alpha=0.75,label='offset offset by 312')
    
    
    # plt.legend()
    # plt.show
    # print(depthsNULL)
    
    
    'maximum on 1000 trials:'
    n=10**3
    sd_1000=randomNoiseSigma
    
    # percentile999=0+((randomNoiseSigma)*(np.sqrt(2)*special.erfcinv(2*0.001)))
    
    # percentile95=0+((sd_1000)*(np.sqrt(2)*special.erfcinv(2*0.05)))
    
    
    depthTrials1000=np.zeros(n)
    
    # for i in np.arange(n):
    #     brightnessNULL_test=bf.makeNullData(mu,randomNoiseSigma,npoints)
    #     insides,outsides=bf.get_insides(t,offsetPeriod,offset,halfDuration)
    #     depthTrials1000[i]=bf.get_depths(brightnessNULL_test,insides,mu)
    
    # file = open('depthTESTS1000', 'wb')
    # pickle.dump(depthTrials1000, file)
    # file.close()
    
    file = open('depthTESTS1000', 'rb')
    depthTrials1000 = pickle.load(file)
    file.close()
    
    mu=0
    'test number N- how many light curves to test'
    N=1000
    
    sd=10**(-4)
    # x = random.normal(loc=mu, scale=sd, size=(N))
    x=depthTrials1000
    
    plt.hist(x,label='derived depths')
    plt.hist(x)
    plt.axvline(np.percentile(x,99.9),label='expectation on 99.9% percentile',color='red')
    
    # plt.axvline(percentile95,label='expectation on 95% percentile',color='blue')
    # plt.axvline(np.percentile(depthTrials1000,99.9),label='numpy 99.9% percentile',color='green')
    # plt.axvline(np.percentile(depthTrials1000,95),label='numpy 99.9% percentile',color='orange')
    plt.show()
    
    plt.figure()
    
    probabilities=np.linspace(0.0,0.999,50)
    # probabilities=np.asarray([0.999])
    # expectedPercentiles=np.mean(x)+(sd*10**(-2))*(-np.sqrt(2)*special.erfcinv(2*(probabilities)))
    expectedPercentiles=findPercentile(x,probabilities)
    plt.plot(probabilities,expectedPercentiles,'b-',label='expectation on percentile distribution')
    percentileProbs=probabilities*100
    plt.plot(probabilities,np.percentile(x,percentileProbs),'k^',label='numpy percentile')
            
    
    plt.legend()
    
    plt.show()
    


def findPercentile(dataset,percentileVals):
    testSet=np.sort(dataset)
    percentileIndeces=(percentileVals*np.size(testSet)).astype(int)
    print(percentileIndeces)
    percentiles=testSet[percentileIndeces]
    return percentiles


main()
