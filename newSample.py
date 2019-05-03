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
    return 0.5*(-b-quad/a), 0.5*(-b+quad/a)

def main():
    
    np.random.seed(0)
    npoints=101;
    mu=1.0; timeEnd=10;
    sigma= 3*(10**(-4))
    
    (t,brightness,randNoise)=makeNullData(mu,sigma,npoints,timeEnd)
    sigmas=sigma*np.ones(len(brightness))
    tstartIndex=4
    widthPoints=1
    brightness=brightness+randNoise
    
    delta=np.linspace(-0.1,0.1,1000)
    diffLL=computeDeltaLLPerDepth(delta,t,mu,brightness,sigmas,tstartIndex,widthPoints)

    (a,b,c)=calc_parabola_vertex(delta[5], diffLL[5], delta[6], diffLL[6], delta[4], diffLL[4])
    target=2.0
    print(a,b,c)
    intercepts=parabolic_roots(a,b,c,target)

    diffLLTest=computeDeltaLLPerDepth(intercepts,t,mu,brightness,sigmas,tstartIndex,widthPoints)
    print(intercepts)
    print(diffLLTest)
    
    diffLLY=np.ones(len(delta))
    for i in range(len(delta)):
        foo=delta[i]
        diffLLY[i]=a*(foo**2)+b*foo+c 
    
    
    f1=plt.figure(1)
    axes = plt.gca()
    axes.set_ylim([-1.0,16.0])
    plt.title('Delta Ln Likelihood vs. Depth of Box')
    plt.ylabel('diff. ln likelihood')
    plt.xlabel('box depth')
    plt.plot(delta,diffLL,'b-',delta,target*np.ones(len(delta)),delta,diffLLY,'r--')
    

    f2=plt.figure(2)
    plt.title('Brightness v. Time')
    plt.ylabel('brightness')
    plt.xlabel('time')
    plt.plot(t,brightness,'k.')
    plt.show()
    

    plt.show()

    

main()


#
#