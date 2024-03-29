import basicDIFFLLFUNCTIONS as dll
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy as sp

np.random.seed(0)
npoints=1500;
mu=1.0; offsetPeriod=100;offset=0.0006;offsetDuration=10;
sigma= (10**(-4))
sigmas=sigma*np.ones(npoints)
halfDuration=offsetDuration/2

def periodicDataTest():
    (t,brightness)=dll.makePeriodicData(mu,sigma,npoints,100,0.0006,10)
    sigmas=sigma*np.ones(len(brightness))
    plt.figure()
    dll.drawBasicBrightnessPlot(t,brightness,sigmas)
    plt.show()
    
        
def boxSweepPeriodicData0():
    sigmas=sigma*np.ones(npoints)
    (t,brightness)=dll.makePeriodicData(mu,sigma,npoints,offsetPeriod,offset,offsetDuration)
    sigmas=sigma*np.ones(len(brightness))
    
    durations=np.asarray([30,10,3])
    durationIndex=0
    '''for each duration'''
    '''filling out depth and diffLL values per depth for all start points'''
    while durationIndex<(len(durations)):
        plt.figure()
        dll.drawBasicBrightnessPlot(t,brightness,sigmas)
        plt.title('BOX MODEL DELTALL: depth=difference between mu, mean of points inside box, duration = '+str(durations[durationIndex])+', $\sigma$=10^-4, gaussian white noise')
        
        duration=durations[durationIndex]
        i=0
        '''diffLL for each start point'''
        while i<len(brightness)-duration:
            depthCurrent=1-np.mean(brightness[i:i+duration])
            i+=1
            plt.plot(dll.boxModel(t,mu,i,i+duration,depthCurrent))
        plt.show()
        durationIndex+=1

            
def boxSweepPeriodicData1():
    (t,brightness)=dll.makePeriodicData(mu,sigma,npoints,offsetPeriod,offset,offsetDuration)
    sigmas=sigma*np.ones(len(brightness))
    
    
    
    durations=np.asarray([30,10,3])
    depthArraysPerDuration = []
    deltaLLArraysPerDuration=[]
    
    durationIndex=0
    '''for each duration'''
    '''filling out depth and diffLL values per depth for all start points'''
    while durationIndex<(len(durations)):
#        f1=plt.figure(durationIndex+1)
#        drawBasicBrightnessPlot(t,brightness,sigmas)
#        plt.title('BOX MODEL DELTALL: depth=difference between mu, mean of points inside box, duration = '+str(durations[durationIndex])+', $\sigma$=10^-4, gaussian white noise')
        
        duration=durations[durationIndex]
        currentDepthArray=np.zeros(npoints-duration)
        currentDiffLL=np.zeros(npoints-duration)
        i=0
        '''diffLL for each start point'''
        while i<len(brightness)-duration:
            depthCurrent=1-np.mean(brightness[i:i+duration])
            currentDepthArray[i]=depthCurrent
            currentDiffLL[i]=dll.computeDeltaLL(depthCurrent,mu,brightness,sigmas,i,duration)
            i+=1
            
        depthArraysPerDuration.append(currentDepthArray)
        deltaLLArraysPerDuration.append(currentDiffLL)
    
#        plt.show()
        
        durationIndex+=1
        
    depthArraysPerDuration = np.asarray(depthArraysPerDuration)
    deltaLLArraysPerDuration=np.asarray(deltaLLArraysPerDuration)
    '''now we have, for each duration, an array of n-duration diffLL and depth values. (n-duration) x 3'''
#    perc = np.array([25,75])
    np.set_printoptions(precision=2)

    for i in range(len(depthArraysPerDuration)):
        fig, axs = plt.subplots(2, 1)
        (ax1), (ax2)= axs
        ax1.hist(deltaLLArraysPerDuration[i],bins=200)
        dll.drawBasicBrightnessSubplot(t,brightness,sigmas,ax2)

        fig.suptitle('$\Delta$ ln likelihood output for duration value='+str(durations[i]))
        
        mean=np.mean(deltaLLArraysPerDuration[i])
        median=np.median(deltaLLArraysPerDuration[i])
        
        mode_info=stats.mode(deltaLLArraysPerDuration[i])
        mode= mode_info[0]
        maxVal=np.max(deltaLLArraysPerDuration[i])
        minVal=np.min(deltaLLArraysPerDuration[i])

        maxDiffLLValIndex=np.argmax(deltaLLArraysPerDuration[i])
        minDiffLLValIndex=np.argmin(deltaLLArraysPerDuration[i])

        ax1.axvline(mean, color='k', linestyle='dashed', linewidth=1, label='first')
        ax1.axvline(median, color='b', linestyle='dashed', linewidth=1)
        ax1.axvline(mode, color='g', linestyle='dashed', linewidth=1)
        ax1.axvline(maxVal, color='y', linestyle='dashed', linewidth=1)
        ax1.axvline(minVal, color='r', linestyle='dashed', linewidth=1)


        min_ylim, max_ylim = plt.ylim()
        ax1.text(depthArraysPerDuration[i].mean()*1.1, max_ylim*0.7, 'Mean: {:.6f}'.format(depthArraysPerDuration[i].mean()))
        ax1.legend({'Mean diffLL: '+str(mean)+' ':mean,'Median diffLL: '+str(mean):median,'Mode diffLL: '+str(mode)+' ':mode,'Max diffLL value: '+str(maxVal)+' ':maxVal,'Min diffLL value: '+str(minVal)+' ':minVal})
        
        maxBox=dll.boxModel(t,mu,maxDiffLLValIndex,maxDiffLLValIndex+durations[i],(depthArraysPerDuration[i][maxDiffLLValIndex]))
        minBox=dll.boxModel(t,mu,minDiffLLValIndex,minDiffLLValIndex+durations[i],(depthArraysPerDuration[i][minDiffLLValIndex]))
        ax2.plot(t,maxBox,color='y',linewidth=1)
        ax2.plot(t,minBox,color='r',linewidth=1)
        
        ax2.legend({'Max Diff LL, occurring at t= '+str(maxDiffLLValIndex)+', with depth = '+str(depthArraysPerDuration[i][maxDiffLLValIndex])+' ':maxBox,'Min Diff LL, occurring at t= '+str(minDiffLLValIndex)+', with depth = '+str(depthArraysPerDuration[i][minDiffLLValIndex])+' ':minBox})

        plt.show()
        
        '''NOW PRINT DEPTH VALUES PER DURATION'''
        
        plt.figure()
        plt.hist(depthArraysPerDuration[i],bins=200,density=True)
        mean=np.mean(depthArraysPerDuration[i])
        median=np.median(depthArraysPerDuration[i])
        
        mode_info=stats.mode(depthArraysPerDuration[i])
        mode= mode_info[0]
        maxVal=np.max(depthArraysPerDuration[i])
        minVal=np.min(depthArraysPerDuration[i])

        plt.axvline(mean, color='k', linestyle='dashed', linewidth=1, label='first')
        plt.axvline(median, color='r', linestyle='dashed', linewidth=1)
        plt.axvline(mode, color='g', linestyle='dashed', linewidth=1)
        plt.axvline(maxVal, color='y', linestyle='dashed', linewidth=1)
        plt.axvline(minVal, color='b', linestyle='dashed', linewidth=1)
        min_ylim, max_ylim = plt.ylim()
        plt.text(depthArraysPerDuration[i].mean()*1.1, max_ylim*0.7, 'Mean: {:.6f}'.format(depthArraysPerDuration[i].mean()))
        plt.legend({'Mean depth: '+str(mean)+' ':mean,'Median depth: '+str(mean):median,'Mode depth: '+str(mode)+' ':mode,'Max depth value: '+str(maxVal)+' ':maxVal,'Min depth value: '+str(minVal)+' ':minVal})
        plt.title('Depths for duration value='+str(durations[i]))
        left, right = plt.xlim()
        
        mean = 0.0 
        std = sigma*np.sqrt(1/durations[i])

        y = sp.stats.norm.pdf(np.linspace(left,right),mean,std)

        
        plt.plot(np.linspace(left,right),y)
        
        mean=(offset*offsetDuration)/durations[i]
        std = np.sqrt(std**2+(offset*np.sqrt(1/(offsetDuration+durations[i])))**2)
#
        y = sp.stats.norm.pdf(np.linspace(left,right),mean,std)
#
#        
        plt.plot(np.linspace(left,right),y)
#
        
        plt.show()
        
    print('end')

def comparisonMatrixMeanDepth():
    sigmas=sigma*np.ones(npoints)
    (t,brightness)=dll.makePeriodicData(mu,sigma,npoints,offsetPeriod,offset,offsetDuration)
    sigmas=sigma*np.ones(len(brightness))
    
    periodLower=offsetPeriod-1; periodUpper=offsetPeriod+1;
    durationLower=offsetDuration-1; durationUpper=offsetDuration+1;  
    
    durations=np.asarray([durationLower,offsetDuration,durationUpper])
    periods=np.asarray([periodLower,offsetPeriod,periodUpper])
    diffLLComparison = (10**6)*np.ones((3,3))
    indecesComparison = np.ones((3,3))
    depthsComparison=np.ones((3,3))


    periodIndex=0
    while periodIndex<len(periods):
        periodCurrent=periods[periodIndex]
        durationIndex=0
        while durationIndex<len(durations):
            durationCurrent=durations[durationIndex]
            boxINDEX=np.arange(periodCurrent)
            for i in boxINDEX:
                deltaLL=dll.computeDeltaLLPeriodic(periodCurrent,mu,i,brightness,sigmas,durationCurrent)
                if deltaLL<diffLLComparison[periodIndex][durationIndex]:
                    diffLLComparison[periodIndex][durationIndex]=deltaLL
                    indecesComparison[periodIndex][durationIndex]=i
#                    depthsComparison[periodIndex][durationIndex]=depthCurrent
            durationIndex+=1
#            print(periodCurrent,durationCurrent)
        periodIndex+=1
    indecesComparison=indecesComparison.astype(int)
    print('\n diffLL: \n',diffLLComparison,
          '\n t start points: \n',indecesComparison,
          '\n depth= mean of points inside box: \n',depthsComparison)
#    minIndex=np.unravel_index(diffLLComparison.argmin(), diffLLComparison.shape)
#    print(minIndex)
    drawComparisonMatrixMeanDepth(durations,periods,indecesComparison,diffLLComparison,depthsComparison)

    
def drawComparisonMatrixMeanDepth(durations,periods,indecesComparison,diffLLComparison,depthsComparison):
    fig, axs = plt.subplots(3, 3)
    sigmas=sigma*np.ones(npoints)
    (t,brightness)=dll.makePeriodicData(mu,sigma,npoints,offsetPeriod,offset,offsetDuration)
    sigmas=sigma*np.ones(len(brightness))


    (ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9) = axs
    fig.suptitle('Testing statistics of BLS duration, BLS period, for periodic data of offsetPeriod= '+str(offsetPeriod)+' and offsetDuration= '+str(offsetDuration))
    axes=axs.flat
    
    for ax in axes:
        ax.errorbar(t,brightness, yerr=sigmas,fmt='ko',markersize=0.7,ecolor='blue')
        ax.label_outer()

    'rows'
    for x in range(0, axs.shape[0]):
        periodCurrent=periods[x]
        'cols'
        for y in range(0, axs.shape[1]):
            durationCurrent=durations[y]
#            depthCurrent=depthsComparison[x][y]
            startPoint=indecesComparison[x][y]
#            print(periodCurrent,durationCurrent)
            axs[x,y].plot(dll.periodicBoxModelAlterableDepth(brightness,t,mu,startPoint,offsetPeriod,durationCurrent),'yellow')
            axs[x,y].title.set_text('BLS period= '+str(periodCurrent)+' and BLS duration= '+str(durationCurrent))
            axs[x,y].text(0.95, 0.01, 'diffLL: '+str(diffLLComparison[x][y])+', start point: '+str(indecesComparison[x][y]),
            verticalalignment='bottom', horizontalalignment='right',transform=axs[x][y].transAxes,fontsize=10)


def comparisonMatrix():
    sigmas=sigma*np.ones(npoints)
    (t,brightness)=dll.makePeriodicData(mu,sigma,npoints,offsetPeriod,offset,offsetDuration)
    sigmas=sigma*np.ones(len(brightness))
    
    periodLower=offsetPeriod-1; periodUpper=offsetPeriod+1;
    durationLower=offsetDuration-1; durationUpper=offsetDuration+1;  
    
    durations=np.asarray([durationLower,offsetDuration,durationUpper])
    periods=np.asarray([periodLower,offsetPeriod,periodUpper])
    diffLLComparison = np.zeros((3,3))
    indecesComparison = np.ones((3,3))
    depthsComparison=np.ones((3,3))


    periodIndex=0
    while periodIndex<len(periods):
        periodCurrent=periods[periodIndex]
        durationIndex=0
        while durationIndex<len(durations):
            durationCurrent=durations[durationIndex]
            boxINDEX=np.arange(periodCurrent-1)
            for i in boxINDEX:
                depthCurrent=mu-np.mean(brightness[i:i+durationCurrent]) #hogg would like to fix
                deltaLL=dll.computeDeltaLLPeriodic(periodCurrent,mu,i,brightness,sigmas,durationCurrent)
                if deltaLL>diffLLComparison[periodIndex][durationIndex]:
                    diffLLComparison[periodIndex][durationIndex]=deltaLL
                    indecesComparison[periodIndex][durationIndex]=i
                    depthsComparison[periodIndex][durationIndex]=depthCurrent
            durationIndex+=1
        periodIndex+=1
    indecesComparison=indecesComparison.astype(int)
    print('\n diffLL: \n',diffLLComparison,
          '\n t start points: \n',indecesComparison,
          '\n depth= mean of points inside box: \n',depthsComparison)
#    minIndex=np.unravel_index(diffLLComparison.argmin(), diffLLComparison.shape)
#    print(minIndex)
    drawComparisonMatrix(durations,periods,indecesComparison,diffLLComparison,depthsComparison)

    
    
    
def drawComparisonMatrix(durations,periods,indecesComparison,diffLLComparison,depthsComparison):
    fig, axs = plt.subplots(3, 3)
    sigmas=sigma*np.ones(npoints)
    (t,brightness)=dll.makePeriodicData(mu,sigma,npoints,offsetPeriod,offset,offsetDuration)
    sigmas=sigma*np.ones(len(brightness))


    (ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9) = axs
    fig.suptitle('Testing statistics of BLS duration, BLS period, for periodic data of offsetPeriod= '+str(offsetPeriod)+' and offsetDuration= '+str(offsetDuration))
    axes=axs.flat
    
    for ax in axes:
        ax.errorbar(t,brightness, yerr=sigmas,fmt='ko',markersize=0.7,ecolor='blue')
        ax.label_outer()

    'rows'
    for x in range(0, axs.shape[0]):
        periodCurrent=periods[x]
        'cols'
        for y in range(0, axs.shape[1]):
            durationCurrent=durations[y]
            depthCurrent=depthsComparison[x][y]
            startPoint=indecesComparison[x][y]
#            print(periodCurrent,durationCurrent)
            axs[x,y].plot(dll.periodicBoxModel(t,mu,startPoint,offsetPeriod,durationCurrent,depthCurrent),'yellow')
            axs[x,y].title.set_text('BLS period= '+str(periodCurrent)+' and BLS duration= '+str(durationCurrent))
            axs[x,y].text(0.95, 0.01, 'diffLL: '+str(diffLLComparison[x][y])+', start point: '+str(indecesComparison[x][y]),
            verticalalignment='bottom', horizontalalignment='right',transform=axs[x][y].transAxes,fontsize=10)

