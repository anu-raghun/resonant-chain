import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

'https://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points'
def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
    denom = (x1-x2) * (x1-x3) * (x2-x3)
    A     = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
    B     = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
    C     = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom;
    return A,B,C


#def llBox(t,mu,brightness,noise,tstart,tend,depth):
#    ll=0
#    for i in range(len(noise)):
#        if (t[i]<=tend and t[i]>=tstart):
#            muNew=mu-depth
#            ll+=-((brightness[i]-muNew)**2)/(2*(noise[i]**2))-(np.log(2*np.pi*(noise[i])**2)/2)
#        else:
#            ll+=-((brightness[i]-mu)**2)/(2*(noise[i]**2))-(np.log(2*np.pi*(noise[i])**2)/2)
#    return ll
#
#def llNull(t,mu,brightness,noise):
#    ll=0
#    for i in range(len(noise)):
#        ll+=-((brightness[i]-mu)**2)/(2*(noise[i]**2))-(np.log(2*np.pi*(noise[i])**2)/2)
#    return ll
        

def makeNullData(mu,sd,nPoints,timeEnd):
    t = np.arange(nPoints)
    brightness = mu * np.ones(nPoints)
    randNoise = np.random.normal(0,sd,nPoints)
    return(t,brightness,randNoise)

#def computeDeltaLLPerDelta(delta,t,mu,brightness,noise,t1,t2):
#    deltaLL=np.zeros(len(delta))
#    for i in range(len(delta)):
#        ll1=llNull(t,mu,brightness,noise)
#        ll2=llBox(t,mu,brightness,noise,t1,t2,delta[i])
#        deltaLL[i]=ll1-ll2
#    return deltaLL

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
    
    

#def computeDeltaLLPerPeriod(delta,t,mu,brightness,noise,t1,periods):
#    deltaLL=np.zeros(len(periods))
#    for i in range(len(periods)):
#        ll1=llNull(t,mu,brightness,noise)
#        t2=t1+periods[i]
#        ll2=llBox(t,mu,brightness,noise,t1,t2,delta)
#        deltaLL[i]=ll1-ll2
#    return deltaLL

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

def f(x):
    return 1106331.2445340601*(x**2)+((0.0)*x)-2.0

def main():
    
    np.random.seed(0)
    npoints=11;
    mu=1.0; timeEnd=10;
    sigma= 3*(10**(-4))
    
    (t,brightness,randNoise)=makeNullData(mu,sigma,npoints,timeEnd)
    tstartIndex=3
    widthPoints=1
    
    delta=np.linspace(-0.1,0.1,npoints)
    diffLL=computeDeltaLLPerDepth(delta,t,mu,brightness,randNoise,tstartIndex,widthPoints)
#
#
    (a,b,c)=calc_parabola_vertex(delta[5], diffLL[5], delta[6], diffLL[6], delta[4], diffLL[4])
    print('parabolic coefficients: ',a,b,c)


    diffLLy=[]
    target=(2.0)*np.ones(len(diffLL))
    
    
    for x in range(len(delta)):
        deltaCurrent=delta[x]
        y=(a*(deltaCurrent**2))+(b*deltaCurrent)+c
        diffLLy.append(y)
    guess=(0.001)
    
    
    x=fsolve(f,guess)
    deltaTest=[x,9.24035e-05]
    print('calculated value of delta vs. graphically obtained value: ',deltaTest)
    diffLLTest=computeDeltaLLPerDepth(deltaTest,t,mu,brightness,randNoise,tstartIndex,widthPoints)
    print('results of plugging the values into the diffLL function: ',diffLLTest)
    
    f1=plt.figure(1)
    
    axes = plt.gca()
    axes.set_ylim([-1.0,16.0])
    plt.title('Delta Ln Likelihood vs. Depth of Box')
    plt.ylabel('diff. ln likelihood')
    plt.xlabel('box depth')
    plt.plot(delta,diffLL,'b--',x,diffLLTest[0],'.k',delta,target)
    
    
    a=np.where(diffLL<= 10)
    print(diffLL[a[0]],'values of diffLL<=10')
    
    
    f2=plt.figure(2)
    plt.title('Delta Ln Likelihood vs. Depth of Box')
    plt.ylabel('diff ln likelihood')
    plt.xlabel('box depth')
    plt.plot(delta,diffLLy,'c-.',delta,target,'r')
    plt.show()
    
    
    
    B = np.where(diffLL<=0.0)
    print(diffLL[B[0]],'where diffLL is leq 0')

#    plt.plot(delta[idx], target[idx], 'ro')
    plt.show()

##    
    # -1*10^ to 10 delta LL plot
#    
    #5 points, 100 points, 1000 points
      
    

main()


#
#