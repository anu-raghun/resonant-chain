import basicPeriodicFunctions as bf
#import basicDIFFLLFUNCTIONS as dll
import numpy as np
import matplotlib.pyplot as plt
import periodicTestRunsMod as ptestmod
from scipy import stats
#import scipy as sp
from scipy.special import erfcinv
import pickle
import astropy.timeseries
import pandas as pd
import time




np.random.seed(0)
trange=[0,1000.7]; 
#npoints=trange[1]*30
npoints=50000
mu=1.0; offsetPeriod=18.767; offset=13.2; transitDuration=0.525;
randomNoiseSigma= (10**(-4))
halfDuration=transitDuration/2
transitDepth=0.0002
delta=0.005

npoints=int(npoints)

t = np.arange(trange[0],trange[1],delta)
sigmas=randomNoiseSigma*np.ones(len(t))
insides,outsides=bf.get_insides(t,offsetPeriod,offset,halfDuration)
(brightness)=bf.makePeriodicData(mu,randomNoiseSigma,len(t),offsetPeriod,offset,insides,transitDepth)
#plt.plot(t,brightness,'.k')
#plt.plot(t,bf.periodicBoxModelAlterableDepth(t,trange,mu,bf.get_depths(brightness,insides,mu),insides),'green')

print('data has a transit of period '+str(offsetPeriod)+'  and offset '+str(offset)+', half-duration= '+str(halfDuration)+'\n')
    
#offsets=np.arange(0,offsetPeriod,transitDuration/3)
#periods=[offsetPeriod]
#ptestmod.iteratingoverOFFSETS(t,brightness,sigmas,halfDuration,trange,periods,'\n offsets from 0 to period, period=actual period',transitDuration/3)
#print('\n')
##
nullBrightness=bf.makeNullData(mu,randomNoiseSigma,len(t))
#ptestmod.iteratingoverOFFSETS(t,nullBrightness,sigmas,halfDuration,trange,offsets,periods,'\n null data trial, same params as above')
#
#print('\n')
#
#start = time.time()

#lnPeriods=np.arange(np.log(16),np.log(20),transitDuration/trange[1])
#periods=np.exp(lnPeriods)
##periods=[offsetPeriod]
#ptestmod.iteratingoverOFFSETS(t,brightness,sigmas,halfDuration,trange,periods,'\n offsets from 0 to period, periods from 0 to 50.425',transitDuration/3)
#plt.plot(t,brightness,'k.')
#end = time.time()
#print(end - start)
#print('\n')
#
#ptestmod.iteratingoverOFFSETS(t,nullBrightness,sigmas,halfDuration,trange,offsets,periods,'\n null data trial as above')

def mainshit():
    '''DO SOME RATIONAL PERIODS
    THEN DO SOME IRRATIONAL PERIODS
    1/E 1/PI, E, PI?
    record: max depth and max delta ll
    also the min depth
    do it on null data too'''
    
    samplePeriod=offsetPeriod*2
    periods=[(1/3)*samplePeriod,(2/3)*samplePeriod,(1/2)*samplePeriod,samplePeriod,(1/np.pi)*samplePeriod,(np.pi)*samplePeriod,(1/np.e)*samplePeriod,(np.e)*samplePeriod]
    periods=np.asarray(periods)
    periods=np.sort(periods)
    
    infoPERPERIOD=np.outer(np.ones(len(periods)),np.zeros(5))
    
    #plt.figure()
    #NHYP=(transitDuration/offsetPeriod)
    #plt.plot(NHYP, special.erf(NHYP))
    #plt.show()
    
    i=0
    while i<len(periods):
        label='\n testing period= '+str(periods[i])+'\n'
        infoPERPERIOD[i][:]=ptestmod.iteratingoverOFFSETS(t,brightness,sigmas,halfDuration,trange,[periods[i]],label,transitDuration/3)
        i+=1
    i=0
    infoPERPERIODNULL=np.ones_like(infoPERPERIOD)
    print(infoPERPERIODNULL.shape)
    while i<len(periods):
        label='\n null data! testing period= '+str(periods[i])+'\n'
        infoPERPERIODNULL[i][:]=ptestmod.iteratingoverOFFSETS(t,nullBrightness,sigmas,halfDuration,trange,[periods[i]],label,transitDuration/3)
        i+=1
    
    fig, axs = plt.subplots(3,1)
    (ax1), (ax2), (ax3)= axs
    
    plt.xlabel("ln box model periods")
     
    ax1.plot(np.log(periods),np.log(np.absolute(infoPERPERIOD[:,0])),'ko',label='data with signal')
    ax1.plot(np.log(periods),np.log(infoPERPERIODNULL[:,0]),'bo',label='null data')
    ax1.set(ylabel='ln max diffLL')
    ax1.legend(loc="upper right")
    
    ax2.plot(np.log(periods),np.log(np.absolute(infoPERPERIOD[:,1])),'ko',label='data with signal')
    ax2.plot(np.log(periods),np.log(np.absolute(infoPERPERIODNULL[:,1])),'bo',label='null data')
    ax2.set(ylabel='ln |max depth|')
    ax2.legend(loc="upper right")
    
    ax3.plot(np.log(periods),np.log(np.absolute(infoPERPERIOD[:,4])),'ko',label='data with signal')
    ax3.plot(np.log(periods),np.log(np.absolute(infoPERPERIODNULL[:,4])),'bo',label='null data')
    ax3.set(ylabel='ln |min depth|')
    ax3.legend(loc="upper right")


def depthASFUNCTIONOFPERIOD():
    periodIndex=0
    while periodIndex<len(periods):
        currentPeriod=periods[periodIndex]
        depthsPerPeriod[periodIndex]=ptestmod.extremeDEPTHS(t,brightness,sigmas,halfDuration,currentPeriod,3*10**(-4))
        depthsPerPeriodNull[periodIndex]=ptestmod.extremeDEPTHS(t,nullBrightness,sigmas,halfDuration,currentPeriod,3*10**(-4))
        periodIndex+=1
        
    file = open('depthasfunctionofperiod', 'wb')
    pickle.dump(depthsPerPeriod, file)
    file.close()
    file = open('depthasfunctionofperiodNULL','wb')
    pickle.dump(depthsPerPeriodNull, file)
    file.close()


def overlayVerticalLines():
    periods=integerMultiples*samplePeriod
    i=0

    for period in periods:
        plt.axvline(period, linewidth=1, alpha= 0.2)
        plt.text(x=period, y=(np.random.rand()+0.05)*0.000125, s=str('%.2f' % integerMultiples[i]), alpha=0.7, rotation=30, color='#334f8d')       
        i+=1
        
    plt.axvline((1/np.e)*samplePeriod, linewidth=1, alpha= 0.2, color='r')
    plt.text(x=(1/np.e)*samplePeriod, y=0.00012, s=str('1/e'), alpha=0.7, color='r')
    plt.axvline((1/np.pi)*samplePeriod, linewidth=1, alpha= 0.2, color='r')
    plt.text(x=(1/np.pi)*samplePeriod, y=0.00012, s=str('1/pi'), alpha=0.7, color='r')
    plt.axvline((1./np.sqrt(2))*samplePeriod, linewidth=1, alpha= 0.2, color='r')
    plt.text(x=(1./np.sqrt(2))*samplePeriod, y=0.00012, s=str('1/$\sqrt{2}$'), alpha=0.7, color='r')

    plt.axvline((np.e)*samplePeriod, linewidth=1, alpha= 0.2, color='r')
    plt.text(x=(np.e)*samplePeriod, y=0.00012, s=str('e'), alpha=0.7, color='r')
    plt.axvline((np.pi)*samplePeriod, linewidth=1, alpha= 0.2, color='r')
    plt.text(x=(np.pi)*samplePeriod, y=0.00012, s=str('pi'), alpha=0.7, color='r')
    plt.axvline((np.sqrt(2))*samplePeriod, linewidth=1, alpha= 0.2, color='r')
    plt.text(x=(np.sqrt(2))*samplePeriod, y=0.00012, s=str('$\sqrt{2}$'), alpha=0.7, color='r')

def get_all_pairs(nmax):
  """
  generate all co-prime pairs by recursion
  """
  pairs = [ ]
  traverse(pairs, nmax, 2, 1)
  traverse(pairs, nmax, 3, 1)
  return pairs

def traverse(pairs, nmax, m, n):
  if m <= nmax and n <= nmax:
    pairs.append((m, n))
    traverse(pairs, nmax, 2 * m - n, m)
    traverse(pairs, nmax, 2 * m + n, m)
    traverse(pairs, nmax, m + 2 * n, n)        
    
def analyzeData(fineRESULTS,fineNULLRESULTS,fineNEGATIVERESULTS):    
    # file = open('depthasfunctionofperiod', 'rb')
    # depthsPerPeriod = pickle.load(file)
    # file.close()

    # file = open('depthasfunctionofperiodNULL', 'rb')
    # depthsPerPeriodNull = pickle.load(file)
    # file.close()


    # plt.plot(periods,depthsPerPeriod[:,0],'ko',label='custom depth function- periodic data -max depth')
    # plt.plot(periods,depthsPerPeriod[:,1],'bo',label='custom depth function- periodic data -min depth')
    
    # plt.plot(periods,depthsPerPeriodNull[:,0],'kv',label='custom depth function- null data- max depth')
    # plt.plot(periods,depthsPerPeriodNull[:,1],'bv',label='custom depth function- null data- min depth')

    plt.xlabel('period')
    plt.semilogx()
    plt.ylabel('depths')
    plt.title('optimal depth as a function of period. Positive injection. Fiducial Period = 1/2 injection period.')
    expectation=returnExpectation(finePeriods)
    
    # plt.axhline(fineNULLRESULTS.depth.max(),label='max depth output from null data',alpha=0.2,color='red')
    # [upper,lower]=returnBoundsOnData(fineRESULTS.depth,fineNULLRESULTS.depth,10,1,expectation)
    # print('bound coefficients: ',upper,lower)
    
    plt.plot(finePeriods,0.8*expectation,'-',color="black",label='lower bound on positive expectation',alpha=0.5,)
    plt.plot(finePeriods,2*expectation,'-',color="black",label='upper bound on positive expectation',alpha=1,)

    # [upperNEGATIVE,lowerNEGATIVE]=returnBoundsOnData(fineNEGATIVERESULTS.depth,-1*fineNEGATIVERESULTS.depth,10,-2,expectation)
    plt.plot(finePeriods,-0.8*expectation,'-',color="green",label='upper bound on negative expectation',alpha=0.5,)
    # print('negative bound coefficients: ', lowerNEGATIVE)

    overlayVerticalLines()
    plt.legend()
    
    plt.figure()
    plt.plot(finePeriods,fineRESULTS.depth/expectation,'--',label='finely gridded periods, run on periodic data',alpha=0.5)
    plt.plot(finePeriods,fineNULLRESULTS.depth/expectation,'--',label='finely gridded periods, run on null data',alpha=0.5)
    plt.plot(finePeriods,-1* fineNEGATIVERESULTS.depth/expectation,'--',label='finely gridded periods, run on negative data',alpha=0.5)


def returnExpectation(periods):
    return np.sqrt((2*(randomNoiseSigma**2)*periods*delta)/(halfDuration*trange[1]))*erfcinv(0.1*halfDuration/periods)

    
samplePeriod=1/2*offsetPeriod
foo = get_all_pairs(7)
foo.sort()
integerMultiples=np.array([1,1])
for f in foo: integerMultiples=np.append(integerMultiples,[f[0] / f[1], f[1] / f[0]])
periods=np.append(integerMultiples,[1/np.pi,1/np.e,1/np.sqrt(2)])*samplePeriod
periods=np.sort(periods)
depthsPerPeriod=np.outer(np.ones(len(periods)),np.zeros(2))
depthsPerPeriodNull=np.ones_like(depthsPerPeriod)
finePeriods=np.arange(np.min(periods),np.max(periods),3*10**(-4))
  
def runningAstropyModelFinePeriods(methodLabel):
    model = astropy.timeseries.BoxLeastSquares(t, brightness)
    results = model.power(finePeriods,halfDuration,method=methodLabel,oversample=3)
    file = open('depthasfunctionofperiodSLOWfine', 'wb')
    pickle.dump(results, file)
    file.close()

def runningAstropyModelFinePeriodsNULL(methodLabel):
    model = astropy.timeseries.BoxLeastSquares(t, nullBrightness)
    results = model.power(finePeriods,halfDuration,method=methodLabel,oversample=3)
    file = open('depthasfunctionofperiodSLOWNULLfine', 'wb')
    pickle.dump(results, file)
    file.close()

def loadAstropyResultsFINEPERIODS():
    file = open('depthasfunctionofperiodSLOWfine', 'rb')
    slowRESULTS = pickle.load(file)
    file.close()
    return slowRESULTS

def loadAstropyResultsNULLFINEPERIODS():
    file = open('depthasfunctionofperiodSLOWNULLfine', 'rb')
    slowRESULTS = pickle.load(file)
    file.close()
    return slowRESULTS

def runTestsAndPickle():
    model = astropy.timeseries.BoxLeastSquares(t, brightness)
    modelNEGATIVE= astropy.timeseries.BoxLeastSquares(t, -1*brightness)
    modelNULL = astropy.timeseries.BoxLeastSquares(t, nullBrightness)   
    
    # results = model.autopower(2*halfDuration,method='fast',oversample=3)
    # file = open('resultsFastPeriodicDiscreteAUTOPOWER', 'wb')
    # pickle.dump(results, file)
    # file.close()
    # results = modelNULL.autopower(2*halfDuration,method='fast',oversample=3)
    # file = open('resultsFastNullDiscreteAUTOPOWER', 'wb')
    # pickle.dump(results, file)
    # file.close()
    # results = modelNEGATIVE.autopower(2*halfDuration,method='fast',oversample=3)
    # file = open('resultsFastPeriodicDiscreteNEGATIVEAUTOPOWER', 'wb')
    # pickle.dump(results, file)
    # file.close()
    
#    results=modelNULL.autopower(2*halfDuration,method='fast',oversample=3,minimum_period=17,maximum_period=19)
#    periods=modelNULL.autoperiod(2*halfDuration, minimum_period=17, maximum_period=19)
#    file = open('autopowerNULL', 'wb')
#    pickle.dump([results,periods], file)
#    file.close()
#    
    periods=modelNULL.autoperiod(1, minimum_period=5, maximum_period=7)
    file = open('testJAN11', 'wb')
    pickle.dump(periods, file)

    # results = model.autopower(2*halfDuration,method='fast',oversample=3)
    # file = open('autopowerPeriodic', 'wb')
    # pickle.dump(results, file)
    # file.close()
    # results=modelNEGATIVE.autopower(2*halfDuration,method='fast',oversample=3)
    # file = open('autopowerNegative', 'wb')
    # pickle.dump(results, file)
    # file.close()

   
def openPickles():
    file = open('resultsFastPeriodicDiscreteAUTOPOWER', 'rb')
    results = pickle.load(file)
    file.close()
    file = open('resultsFastNullDiscreteAUTOPOWER', 'rb')
    resultsNULL = pickle.load(file)
    file.close()
    file = open('resultsFastPeriodicDiscreteNEGATIVEAUTOPOWER', 'rb')
    resultsNEGATIVE = pickle.load(file)
    file.close()
    file = open('finePeriodsFastNullAUTOPOWER', 'rb')
    fineNULLRESULTS = pickle.load(file)
    file.close()
    file = open('finePeriodsFastPeriodicAUTOPOWER', 'rb')
    fineRESULTS = pickle.load(file)
    file.close()
    file = open('finePeriodsFastNegativeAUTOPOWER', 'rb')
    fineNEGATIVERESULTS = pickle.load(file)
    file.close()
    return [results,resultsNULL,resultsNEGATIVE,fineNULLRESULTS,fineRESULTS,fineNEGATIVERESULTS]


def comparingNull_Negative_Periodic_thresholds():
    # runTestsAndPickle()
    # depthASFUNCTIONOFPERIOD()
    [results,resultsNULL,resultsNEGATIVE,fineNULLRESULTS,fineRESULTS,fineNEGATIVERESULTS]=openPickles()
    
    
    plt.plot(periods,results.depth,'o',label='discrete integer multiple periods tested on periodic data')
    plt.plot(periods,resultsNULL.depth,'v',label='discrete integer multiple periods tested on NULL data')
    plt.plot(periods,-1*resultsNEGATIVE.depth,'x',label='discrete integer multiple periods tested on negative data')
    
    
    plt.plot(finePeriods,fineRESULTS.depth,'--',label='finely gridded periods, run on periodic data',alpha=0.5)
    plt.plot(finePeriods,fineNULLRESULTS.depth,'--',label='finely gridded periods, run on null data',alpha=0.5)
    plt.plot(finePeriods,-1* fineNEGATIVERESULTS.depth,'--',label='finely gridded periods, run on negative data',alpha=0.5)
    
    print(integerMultiples.shape)
    analyzeData(fineRESULTS,fineNULLRESULTS,fineNEGATIVERESULTS)


def generateNULLDataFrame(no_Trials):
    nullDATA=np.zeros([no_Trials,len(t)])
    for i in np.arange(0,no_Trials):
        NULLbrightness_temp=bf.makeNullData(mu,randomNoiseSigma,len(t))
        nullDATA[i,:]=NULLbrightness_temp
    print(nullDATA) 
    file = open('nullDATA', 'wb')
    pickle.dump(nullDATA, file)
    file.close()

def openPickle(name):
    file = open(name, 'rb')
    df = pickle.load(file)
    file.close()
    return df

def calculateFinePeriodsPer(nullARRAY):

    depthResultsFINE=np.ones([len(nullARRAY),len(finePeriods)])
    i=0
    for nullArray in nullARRAY:
        start = time.time()
        modelNULL_temp = astropy.timeseries.BoxLeastSquares(t, nullArray)   
        resultsFINE_temp = modelNULL_temp.power(finePeriods,halfDuration,method='fast',oversample=3)
        depthResultsFINE[i,:]=resultsFINE_temp.depth
        i+=1
        end = time.time()
        print(start-end)
    file = open('depthResultsFINE', 'wb')
    pickle.dump(depthResultsFINE, file)
    file.close()

    
    return depthResultsFINE

def returnBounds(guess,expectation,depths):
    count_vals = sum(np.greater(depths,guess*expectation))
    percentile_val = 100 * (count_vals/len(depths))
    while (percentile_val >=6):
        guess+=0.005
        count_vals = sum(np.greater(depths,guess*expectation))
        percentile_val = 100 * (count_vals/len(depths))
    print(percentile_val,guess)
    return percentile_val,guess
        
        
def plotDataAndAnalyzeThresholds(nullTRIALS_DISCRETE):
    expectation=returnExpectation(periods)
    fig, axs = plt.subplots(2, 1)
    plt.xlabel('period')
    plt.semilogx()
    plt.ylabel('depths')
    plt.title('null trial no. 7')
    axs[0].set_xscale('log')
    axs[0].plot(periods,expectation,'-',alpha=1)

    for i in np.arange(len(nullTRIALS_DISCRETE)):
        discretePeriodDepths=nullTRIALS_DISCRETE[i]
        axs[0].plot(periods,discretePeriodDepths,'.',alpha=0.5)
        (percentile_val,guess)=returnBounds(1,expectation,discretePeriodDepths)
        axs[0].plot(periods,guess*expectation,'-',label='upper bound on expectation for index '+str(i),alpha=1)        
    axs[1].plot(periods,nullTRIALS_DISCRETE[7],'.',alpha=0.5)
    axs[1].plot(periods,expectation,'-',label='expectation',alpha=1)
    
    guess=1
    count_vals = sum(np.greater(nullTRIALS_DISCRETE[7],guess*expectation))
    percentile_val = 100 * (count_vals/len(nullTRIALS_DISCRETE[7]))
    print(percentile_val)
    while (percentile_val >=6):
        guess+=0.005
        count_vals = sum(np.greater(nullTRIALS_DISCRETE[7],guess*expectation))
        percentile_val = 100 * (count_vals/len(nullTRIALS_DISCRETE[7]))
    axs[1].plot(periods,guess*expectation,'-',label='upper bound on expectation- no more than 5% false positive rate, coefficient:'+str(guess),alpha=1)

    plt.legend()

    
    plt.show()


def thresholdREFINE():
    # nullDATA=openPickle('nullDATA')
    runTestsAndPickle()
    nullTRIALS_FINEautopower=openPickle('autopowerNULL')
    periodsAUTOPOWER=nullTRIALS_FINEautopower.period
    
    nullTRIALS_DISCRETE=openPickle('depthResultsDISCRETE')
    nullTRIALS_FINE=openPickle('depthResultsFINE')

    # nullTRIALS_FINE=calculateFinePeriodsPer(nullDATA)
    # percentiles=np.percentile(nullTRIALS_FINEautopower.depth,95.0,axis=0)
    expectation=returnExpectation(periodsAUTOPOWER)
    (nullTRIALS_FINEautopower.period)
    
#    for factor in np.arange(1.0,2.05,0.2):
    factor=1.0
    plt.plot(periodsAUTOPOWER,factor*expectation,'k-')
        
    # plt.plot(nullTRIALS_FINEautopower.period,percentiles,'.')
    # plt.plot(nullTRIALS_FINEautopower.period,np.percentile(nullTRIALS_FINEautopower.depth,95.0,axis=0),'--',alpha=0.5)
    # plt.xlabel('periods')
    # plt.ylabel('95th percentiles')
    
    print(nullTRIALS_FINEautopower.depth.shape)
    print(nullTRIALS_FINEautopower.period.shape)
    print(nullTRIALS_FINEautopower)
    print(periodsAUTOPOWER.shape)
    print(nullTRIALS_DISCRETE.shape)
    print(nullTRIALS_FINE.shape)
    periodsTEST=openPickle('testJAN11')

    print(periodsTEST.shape)
    
    
    # y=np.percentile(nullTRIALS_FINEautopower.depth,95.0,axis=0)
#    y2=np.percentile(nullTRIALS_DISCRETE,95.0,axis=0)
    
#    print('y amount:',y2)
    
    x=np.sqrt(periodsAUTOPOWER)
    degree=3
    coeff=np.polyfit(x,nullTRIALS_FINEautopower.depth,degree)
    threshold=np.polyval(coeff,x)
    plt.plot(x**2,threshold,'b-')
    print(coeff)
    
    
    
    # nullTRIALSRATIO=nullTRIALS_FINEautopower/threshold
    # print(np.percentile(np.max(nullTRIALS_DISCRETE,axis=1),95.0))
    # plotDataAndAnalyzeThresholds(nullTRIALS_FINEautopower)


def paperAnalysis():
    periodic1=bf.makePeriodicData(mu,sd,nPoints,period,offset,insides,transitDepth)
    
    
# comparingNull_Negative_Periodic_thresholds()   
# thresholdREFINE()

paperAnalysis()