import numpy as np
import matplotlib.pyplot as plt
#from scipy import stats

#using periodic data with HALF DURATION ITERANTS

def get_insides(ts,period,toffset,halfduration):
    insides=np.abs(np.mod(ts-toffset,period))<halfduration
    outsides=np.logical_not(insides)
    return(insides,outsides)
    
def get_depths(brightness,insides,outsidesAverage):
    return outsidesAverage-np.mean(brightness[np.array(insides)])

def makePeriodicData(mu,sd,nPoints,period,offset,insides,transitDepth):
    brightness = mu * np.ones(nPoints)
    brightness[np.asarray(insides)]-= transitDepth
#    randNoise = np.random.normal(0,sd,nPoints)
    brightness=brightness+np.random.normal(0,sd,nPoints)
    return brightness

def makeNullData(mu,sd,nPoints):
    brightness = mu * np.ones(nPoints)
    brightness=brightness+np.random.normal(0,sd,nPoints)
    return brightness
    
def periodicBoxModelAlterableDepth(t,trange,mu,depths,insides):
    line=np.ones(len(t))*mu
    line[np.asarray(insides)]-=depths
    return line


def computeDeltaLLPeriodic(mu,insides,depths,sd,brightness):
    JJ=np.array(insides)
    d2=np.sum(-0.5*(brightness[JJ]-(mu-depths))**2/sd[JJ]**2)
    d2Null=np.sum(-0.5*(brightness[JJ]-mu)**2/sd[JJ]**2)
    return d2-d2Null

def drawColorBar(xARRAY,yARRAY,COLORBAR,xLabel,yLabel,colorBarLabel):
    plt.figure()
    plt.scatter(xARRAY.flatten(),yARRAY.flatten(),c=COLORBAR.flatten())
    maxINDEX=np.argmax(COLORBAR)
    randINDEX=np.random.choice(maxINDEX,1)[0]

    plt.plot(xARRAY.flatten()[maxINDEX],yARRAY.flatten()[maxINDEX],'r*')
    plt.plot(xARRAY.flatten()[randINDEX],yARRAY.flatten()[randINDEX],'w*')
    label1='\n ideal model '+colorBarLabel+str(COLORBAR.flatten()[maxINDEX])
    label2='\n random '+colorBarLabel+str(COLORBAR.flatten()[randINDEX])

    plt.title('c= '+colorBarLabel+label1+label2)
    plt.xlabel=xLabel
    plt.ylabel=yLabel
    plt.colorbar()
    plt.show()
    
    return (maxINDEX,randINDEX)
    


def drawBrightnessComparison(periodArrays,offsetArrays,diffLLArrays,depthArrays,maxDLLScatterIndex,randomIndex,halfDuration,t,mu,brightness,sigmas,colorBarLabel,trange):
    fig, axs = plt.subplots(1,2)
    (ax1, ax2) = axs
    fig.suptitle('Maximizing '+colorBarLabel)
    axes=axs.flat
    
    maxINFO='period= '+str(periodArrays.flatten()[maxDLLScatterIndex])+', \n offset= '+str(offsetArrays.flatten()[maxDLLScatterIndex])+', depth= '+str(depthArrays.flatten()[maxDLLScatterIndex])
    randINFO='period= '+str(periodArrays.flatten()[randomIndex])+', \n offset= '+str(offsetArrays.flatten()[randomIndex])+', depth= '+str(depthArrays.flatten()[randomIndex])
    
    ax1.title.set_text(maxINFO)
    ax2.title.set_text(randINFO)

    for ax in axes:
        ax.errorbar(t,brightness, yerr=sigmas,fmt='ko',markersize=0.7,ecolor='blue')
        ax.label_outer()

    insidesMAX,outsidesMAX=get_insides(t,periodArrays.flatten()[maxDLLScatterIndex],offsetArrays.flatten()[maxDLLScatterIndex],halfDuration)
    insidesRANDOM,outsidesRANDOM=get_insides(t,periodArrays.flatten()[randomIndex],offsetArrays.flatten()[randomIndex],halfDuration)

#    print(offsetCurrent,periodCurrent,depthCurrent)
    ax1.plot(t,periodicBoxModelAlterableDepth(t,trange,mu,depthArrays.flatten()[maxDLLScatterIndex],insidesMAX),'red')
    ax2.plot(t,periodicBoxModelAlterableDepth(t,trange,mu,depthArrays.flatten()[randomIndex],insidesRANDOM),'green')


#    indexing+=1
        
    plt.show()
