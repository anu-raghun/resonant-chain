import basicPeriodicFunctions as bf
import basicDIFFLLFUNCTIONS as dll
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec


def comparingPeriodFixedWrongDuration(offsetPeriod,halfDurationWrong,offset,brightness,sigmas,t,trange,halfDuration):  
    period1=offsetPeriod/np.sqrt(2); period2=offsetPeriod*np.sqrt(2);
    
    
    periods=np.asarray([period1,period2,offsetPeriod])
    
    diffLLComparison = np.zeros(3)
    depthsComparison=np.ones(3)

    periodIndex=0
    while periodIndex<len(periods):
        periodCurrent=periods[periodIndex]
        insidesMODEL,outsides=bf.get_insides(t,periodCurrent,offset,halfDurationWrong)
        mu=np.mean(brightness[np.asarray(outsides)])
        depths=bf.get_depths(brightness,insidesMODEL,mu)
        deltaLL=bf.computeDeltaLLPeriodic(mu,insidesMODEL,depths,sigmas,brightness)
        if deltaLL>diffLLComparison[periodIndex]:
            diffLLComparison[periodIndex]=deltaLL
            depthsComparison[periodIndex]=depths
        periodIndex+=1
    print('\n diffLL: \n',diffLLComparison,
          '\n depth= mean of points inside box: \n',depthsComparison)
    drawComparisonMatrixPeriods(periods,diffLLComparison,depthsComparison,offsetPeriod,halfDuration,offset,t,trange,brightness,sigmas,halfDurationWrong)

def drawComparisonMatrixPeriods(periods,diffLLComparison,depthsComparison,offsetPeriod,halfDuration,offset,t,trange,brightness,sigmas,halfDurationWrong):
    fig, axs = plt.subplots(periods.size)
    
    (ax1, ax2, ax3) = axs
    fig.suptitle('Testing statistics of BLS period, half actual '+r'$\delta$'+', for periodic data of offsetPeriod= '+str(offsetPeriod)+' and half duration= '+str(halfDuration))
    axes=axs.flat
    
    for ax in axes:
        ax.errorbar(t,brightness, yerr=sigmas,fmt='wo',markersize=0.7,ecolor='blue')
        ax.label_outer()

    for y in range(0, axs.shape[0]):
        periodCurrent=periods[y]
        insidesMODEL,outsides=bf.get_insides(t,periodCurrent,offset,halfDurationWrong)
        mu=np.mean(brightness[np.asarray(outsides)])
        print(periods[y],depthsComparison[y])
        axs[y].plot(t,bf.periodicBoxModelAlterableDepth(t,trange,mu,depthsComparison[y],insidesMODEL),'red')
        axs[y].title.set_text('BLS period= '+str(periodCurrent)+' and BLS '+r'$\delta$ '+str(halfDurationWrong))
        axs[y].text(0.95, 0.01, 'diffLL: '+str(diffLLComparison[y]),
        verticalalignment='bottom', horizontalalignment='right',transform=axs[y].transAxes,fontsize=10)
        
        
def iteratingoverOFFSETS(t,brightness,sigmas,halfDuration,trange,periods,runDescription,arraySpacing):    
    periodIndex=0
    deltaLLMAX=1
    
    print('\n trial description: '+runDescription)
    
    diffLLPerOffset = np.zeros(len(periods))
    depthsPerOffset = np.zeros(len(periods))
    minDepth=5
    '''for each duration'''
    '''filling out depth and diffLL values per depth'''
    while periodIndex<len(periods):
        periodCurrent=periods[periodIndex]
        offsets=np.arange(0,periodCurrent,arraySpacing/3)
        offsetIndex=0
        while offsetIndex<len(offsets):
            offsetCurrent=offsets[offsetIndex]
            insidesMODEL,outsides=bf.get_insides(t,periodCurrent,offsetCurrent,halfDuration)
            muBOX=np.mean(brightness[np.asarray(outsides)])
            depths=bf.get_depths(brightness,insidesMODEL,muBOX)
            deltaLL=bf.computeDeltaLLPeriodic(muBOX,insidesMODEL,depths,sigmas,brightness)
            diffLLPerOffset[periodIndex]=deltaLL
            depthsPerOffset[periodIndex]=depths
            if (deltaLL>deltaLLMAX):
                deltaLLMAX=deltaLL
                depthMAX=depths
                offsetMAX=offsetCurrent
                periodMAX=periodCurrent
            elif depths<minDepth:
                minDepth=depths
            offsetIndex+=1
        periodIndex+=1
    print('max diffLL: '+str(deltaLLMAX)+'\n depth='+str(depthMAX)+'\n offset='+str(offsetMAX)+'\n period='+str(periodMAX))
    return(deltaLLMAX,depthMAX,offsetMAX,periodMAX,minDepth)

def optimalOFFSET(t,brightness,sigmas,halfDuration,trange,modelPeriod,arraySpacing):
    deltaLLMAX=1
    offsets=np.arange(0,modelPeriod,arraySpacing/3)
    offsetIndex=0
    while offsetIndex<len(offsets):
        offsetCurrent=offsets[offsetIndex]
        insidesMODEL,outsides=bf.get_insides(t,modelPeriod,offsetCurrent,halfDuration)
        muBOX=np.mean(brightness[np.asarray(outsides)])
        depths=bf.get_depths(brightness,insidesMODEL,muBOX)
        deltaLL=bf.computeDeltaLLPeriodic(muBOX,insidesMODEL,depths,sigmas,brightness)
        if (deltaLL>deltaLLMAX):
            offsetMAX=offsetCurrent
            deltaLLMAX=deltaLL
        offsetIndex+=1
    return(offsetMAX) 

def extremeDEPTHS(t,brightness,sigmas,halfDuration,modelPeriod,arraySpacing):
    depthMAX=0
    depthMIN=10
    offsets=np.arange(0,modelPeriod,arraySpacing)
    offsetIndex=0
    while offsetIndex<len(offsets):
        offsetCurrent=offsets[offsetIndex]
        insidesMODEL,outsides=bf.get_insides(t,modelPeriod,offsetCurrent,halfDuration)
        muBOX=np.mean(brightness[np.asarray(outsides)])
        depths=bf.get_depths(brightness,insidesMODEL,muBOX)
        if (depths>depthMAX):
            depthMAX=depths
        elif (depths<depthMIN):
            depthMIN=depths
        offsetIndex+=1
    return(depthMAX,depthMIN)

def drawComparisonOffsetVarying(offsets,ArrayPerOffset,depthMAX,modelMAX,modelMAXOUTSIDE,yLabel,t,trange,brightness,sigmas,offset):
    fig, axs = plt.subplots(2,1)
#    plt.title('Data has real offset= '+str(offset)+', period= '+str(offsetPeriod))
    (ax1), (ax2)= axs

    
    ax1.plot(offsets,ArrayPerOffset,)
    ax1.set(xlabel='Model Offset', ylabel=yLabel)

#    ax1.axhline(mean, color='k', linestyle='dashed', linewidth=1, label='first')
#    ax1.axvline(offsetMAXMODEL, color='y', linestyle='dashed', linewidth=1)
#    ax1.text(mean*1.1, max_ylim*0.7, 'Mean: {:.6f}'.format(mean))

    ax1.legend({'Max Offset Value: '+str(depthMAX)+' ':depthMAX})

    muBOX=np.mean(brightness[np.asarray(modelMAXOUTSIDE)])
    dll.drawBasicBrightnessSubplot(t,brightness,sigmas,ax2)
    ax2.plot(t,bf.periodicBoxModelAlterableDepth(t,trange,muBOX,depthMAX,modelMAX),'red')
    ax2.set(title='ideal model parameters for diffLLMAX') 
    plt.show()



def OffsetsPeriodsOffsetVarying(t,brightness,offsetPeriod,sigmas,halfDuration,mu,offsetData,trange):
    offsets=np.linspace(10,15,num=64)

    lnPeriods=np.linspace(np.log(15),np.log(20),num=128)
    offsetIndex=0
    periodIndex=0
    
    
    depthArrays=np.zeros((len(offsets),len(lnPeriods)))
    diffLLArrays=np.ones_like(depthArrays)
    lnPeriodArrays=np.outer(np.ones(len(offsets)),lnPeriods)
#    lnperiodArrays=np.log(periodArrays)
    offsetArrays=np.outer(offsets,np.ones(len(lnPeriods)))

    print(offsetArrays.shape,depthArrays.shape)
    
    deltaLLMAX=np.zeros_like(lnPeriods)
    depthMAX=np.zeros_like(lnPeriods)
    offsetsMAX=np.zeros_like(lnPeriods)
    periodsMAX=np.zeros_like(lnPeriods)

    randomOffsets=np.zeros_like(lnPeriods)
    randomDeltaLL=np.zeros_like(lnPeriods)
    randomDepths=np.zeros_like(lnPeriods)

    
    while periodIndex<(len(lnPeriods)):
        dLLMAXVAL=0
        offsetIndex=0
        while offsetIndex<(len(offsets)):
            dLLMAXVAL=deltaLLMAX[periodIndex]
            offsetCurrent=offsets[offsetIndex]
            periodCurrent=np.exp(lnPeriods[periodIndex])
            insidesMODEL,outsides=bf.get_insides(t,periodCurrent,offsetCurrent,halfDuration)
            depths=bf.get_depths(brightness,insidesMODEL,np.mean(brightness[np.array(outsides)]))
            deltaLL=bf.computeDeltaLLPeriodic(mu,insidesMODEL,depths,sigmas,brightness)
            diffLLArrays[offsetIndex][periodIndex]=deltaLL
            depthArrays[offsetIndex][periodIndex]=depths
            if (deltaLL>dLLMAXVAL):
                deltaLLMAX[periodIndex]=deltaLL
                depthMAX[periodIndex]=depths
                offsetsMAX[periodIndex]=offsetCurrent
                periodsMAX[periodIndex]=periodCurrent
                dLLMAXVAL=deltaLL
            offsetIndex+=1
        randomOffsets[periodIndex]=offsets[np.random.randint(len(offsets))]
        randomInsides,randomOutsides=bf.get_insides(t,periodCurrent,randomOffsets[periodIndex],halfDuration)
        randomDepths[periodIndex]=bf.get_depths(brightness,randomInsides,np.mean(brightness[np.array(randomOutsides)]))
        randomDeltaLL[periodIndex]=bf.computeDeltaLLPeriodic(mu,randomInsides,randomDepths[periodIndex],sigmas,brightness)
        
        periodIndex+=1
         
    
    maxDLLScatterIndex,randomIndex=bf.drawColorBar(lnPeriodArrays,offsetArrays,diffLLArrays,'lnPeriod','offset','delta ln likelihood')
    colorBarLabel='delta ln likelihood.'
    bf.drawBrightnessComparison(lnPeriodArrays,offsetArrays,diffLLArrays,depthArrays,maxDLLScatterIndex,randomIndex,halfDuration,t,mu,brightness,sigmas,colorBarLabel,trange)
#    maxDLLScatterIndex,randomIndex=bf.drawColorBar(lnperiodArrays,offsetArrays,depthArrays,'lnPeriod','offset','depth')
#    colorBarLabel='depth.'
#    bf.drawBrightnessComparison(periodArrays,offsetArrays,diffLLArrays,depthArrays,maxDLLScatterIndex,randomIndex,halfDuration,t,mu,brightness,sigmas,colorBarLabel,trange)

## to address:  period = right period. and then plot likelihood as a function of offset. depth as a function of offset. same plot at wrong period. same plots.
## how much more confident are you knowing the period of a planet. how about searching over period and offset vs just offset. 
## do a fine grid of offset.
    #do grids of different numbers of period and offsets.

def fixedPeriodVaryingOffsetComparison(brightness,sigmas,period,halfDuration,t):
    offsets=np.linspace(0,period,num=4056)
    deltaLLs=np.zeros_like(offsets)
    depths=np.zeros_like(offsets)
    index=0
    while index<len(offsets):
        offset=offsets[index]
        insidesMODEL,outsides=bf.get_insides(t,period,offset,halfDuration)
        depths[index]=bf.get_depths(brightness,insidesMODEL,np.mean(brightness[np.array(outsides)]))
        deltaLLs[index]=bf.computeDeltaLLPeriodic(np.mean(brightness[np.asarray(outsides)]),insidesMODEL,depths[index],sigmas,brightness)
        index+=1
    
    fig, axs = plt.subplots(1, 2)
    axes=axs.flat
    
    fig.suptitle('period= '+str(period))
    axes[0].plot(offsets,deltaLLs)
    axs[0].set(xlabel='offsets', ylabel='deltaLLs')
    
    axes[1].plot(offsets,depths)
    axs[1].set(xlabel='offsets', ylabel='depths')

    

    
def drawComparisonMatrix(trange,deltaLLMAX,depthMAX,offsetsMAX,t,brightness,sigmas,offsetPeriod,halfDuration,offsetData,periods,offsets,mu,randomOffsets,randomDeltaLL,randomDepths):
    fig, axs = plt.subplots(3, 3)
    (ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9) = axs

    fig.suptitle('Testing statistics of BLS duration, BLS period, for periodic data of offset= '+str(offsetData)+', halfDuration= '+str(halfDuration)+', and period= '+str(offsetPeriod)+'. \n Iterating offsets and periods over a range from 0 to twice their actual value. \n Optimized model in green and random model in red.')
    axes=axs.flat
    
    for ax in axes:
        ax.errorbar(t,brightness, yerr=sigmas,fmt='ko',markersize=0.7,ecolor='blue')
        ax.label_outer()

    'rows'
    trackingIndex=-1
    for x in range(0, 3):
        'cols'
        for y in range(0, 3):
            trackingIndex+=1
            periodCurrent=periods[trackingIndex]
            depthCurrent=depthMAX[trackingIndex]
            offsetCurrent=offsetsMAX[trackingIndex]
            
            
            insidesCurrent,outsidesCurrent=bf.get_insides(t,periodCurrent,offsetCurrent,halfDuration)
            
            axs[x,y].plot(t,bf.periodicBoxModelAlterableDepth(t,trange,mu,depthCurrent,insidesCurrent),'green')
            
            randInsides,randOutsides=bf.get_insides(t,periodCurrent,randomOffsets[trackingIndex],halfDuration)
            randDepth=randomDepths[trackingIndex]
            axs[x,y].plot(t,bf.periodicBoxModelAlterableDepth(t,trange,mu,randDepth,randInsides),'red')

            axs[x,y].title.set_text('BLS period= '+str(periodCurrent)+' and BLS offset= '+str(offsetCurrent))
            axs[x,y].text(0.95, 0.01, 'model max diffLL: '+str(deltaLLMAX[trackingIndex]),
            verticalalignment='bottom', horizontalalignment='right',transform=axs[x][y].transAxes,fontsize=10)
            axs[x,y].text(0.95, 0.93, 'random diffLL: '+str(randomDeltaLL[trackingIndex]),
            verticalalignment='bottom', horizontalalignment='right',transform=axs[x][y].transAxes,fontsize=8)
            axs[x,y].text(0.4, 0.93, 'random offset: '+str(randomOffsets[trackingIndex]),
            verticalalignment='bottom', horizontalalignment='right',transform=axs[x][y].transAxes,fontsize=8)


    plt.show()