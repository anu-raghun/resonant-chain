import basicDIFFLLFUNCTIONS as dll
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def boxSweepDataDEPTHSTARTPOINTS():
    np.random.seed(0)
    npoints=1500;
    mu=1.0; timeEnd=10;
    sigma= (10**(-4))
    
    (t,brightness)=dll.makeNullData(mu,sigma,npoints,timeEnd)
    sigmas=sigma*np.ones(len(brightness))
    durations=np.asarray([30,10,3])
    depthArraysPerDuration = []
    deltaLLArraysPerDuration=[]
    
    '''here i am going through the box model for each value and printing depth array and mean'''
    durationIndex=0
    while durationIndex<(len(durations)):
#        f1=plt.figure(durationIndex+1)
#        drawBasicBrightnessPlot(t,brightness,sigmas)
#        plt.title('BOX MODEL DELTALL: depth=difference between mu, mean of points inside box, duration = '+str(durations[durationIndex])+', $\sigma$=10^-4, gaussian white noise')
        
        duration=durations[durationIndex]
        currentDepthArray=np.zeros(npoints-duration)
        currentDiffLL=np.zeros(npoints-duration)
        i=0
        while i<len(brightness)-duration:
            depthCurrent=1-np.mean(brightness[i:i+duration])
            currentDepthArray[i]=depthCurrent
            currentDiffLL[i]=dll.computeDeltaLL(depthCurrent,mu,brightness,sigmas,i,duration)
            
#            plt.plot(boxModel(t,mu,i,i+duration,depthCurrent))
            i+=1
            
        depthArraysPerDuration.append(currentDepthArray)
        deltaLLArraysPerDuration.append(currentDiffLL)
    
#        plt.show()
        
        durationIndex+=1

    depthArraysPerDuration = np.asarray(depthArraysPerDuration)
    print(depthArraysPerDuration[0])
    deltaLLArraysPerDuration=np.asarray(deltaLLArraysPerDuration)
    
#    perc = np.array([25,75])
    np.set_printoptions(precision=2)
    testingARRAY=depthArraysPerDuration

    plotTitle='$\Delta$ ln likelihood output for duration value='
    for i in range(len(depthArraysPerDuration)):
#        plt.figure()
        fig, axs = plt.subplots(2, 1)
        (ax1), (ax2)= axs
        ax1.hist(depthArraysPerDuration[i],bins=200)
        dll.drawBasicBrightnessSubplot(t,brightness,sigmas,ax2)

#        fig, a =  plt.subplots(2,1)
#        fig.text(0.5, 0.04, 'testxaxis', ha='center', va='center')
#        fig.text(0.5, 0.5, 'testxaxis', ha='center', va='center')
#
#        fig.text(0.06, 0.75, 'test y1', ha='center', va='center', rotation='vertical')
#        fig.text(0.06, 0.25, 'test y2', ha='center', va='center', rotation='vertical')
#
#        fig.suptitle('something')
#
##        f2=plt.figure(i+4)
#        a[0][0].set_xlabel('try out')
#        a[0][0].plot.hist(testingARRAY[i],bins=150)
#        a[0][0].title(plotTitle+str(durations[i]))
        
        mean=np.mean(testingARRAY[i])
        median=np.median(testingARRAY[i])
        
        mode_info=stats.mode(testingARRAY[i])
        mode= mode_info[0]
        maxVal=np.max(testingARRAY[i])
        minVal=np.min(testingARRAY[i])
#        maxIndeces.append()
        

        ax1.axvline(mean, color='k', linestyle='dashed', linewidth=1, label='first')
        ax1.axvline(median, color='r', linestyle='dashed', linewidth=1)
        ax1.axvline(mode, color='g', linestyle='dashed', linewidth=1)
        ax1.axvline(maxVal, color='y', linestyle='dashed', linewidth=1)
        ax1.axvline(minVal, color='b', linestyle='dashed', linewidth=1)


        min_ylim, max_ylim = plt.ylim()
        ax1.text(depthArraysPerDuration[i].mean()*1.1, max_ylim*0.7, 'Mean: {:.6f}'.format(depthArraysPerDuration[i].mean()))
        ax1.legend({'Mean depths: '+str(mean)+' ':mean,'Median of depths: '+str(mean):median,'Mode of depths: '+str(mode)+' ':mode,'Max depth value: '+str(maxVal)+' ':maxVal,'Min depth value: '+str(minVal)+' ':minVal})



        plt.show()


        
