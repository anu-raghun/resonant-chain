import numpy as np
import matplotlib.pyplot as plt


def llBox(t,mu,brightness,noise,t1,t2,delta):
    ll=0
    for i in range(len(t)):
        if t[i]<t1:
            ll+=-((brightness[i]-mu)/(2*(noise[i]**2)))-(np.log(2*np.pi*(noise[i]**2))/(2))
        elif t[i]<= t2 and t[i]>=t1:
            muNew=mu-delta
            ll+=-((brightness[i]-muNew)/(2*(noise[i]**2)))-(np.log(2*np.pi*(noise[i]**2))/(2))
        elif (t[i]>t2):
            ll+=-((brightness[i]-mu)/(2*(noise[i]**2)))-(np.log(2*np.pi*(noise[i]**2))/(2))
    return ll

def llNull(t,mu,brightness,noise):
    ll=0
    for i in range(len(t)):
        ll+=-((brightness[i]-mu)/(2*(noise[i]**2)))-(np.log(2*np.pi*(noise[i]**2))/(2))
    return ll

def makeNullData(mu,sd,nPoints,timeEnd):
    t = np.linspace(1, timeEnd, nPoints)
    brightness = mu * np.ones(nPoints)
    randNoise = np.random.normal(0,sd,nPoints)
    return(t,brightness,randNoise)


def computeDeltaLL(delta,t,mu,brightness,noise,t1,t2):
    for i in range(len(delta)):
        ll1=llNull(t,mu,brightness,noise)
        ll2=llBox(t,mu,brightness,noise,t1,t2,delta)
        deltaLL=ll1-ll2
    return deltaLL

    
def main():
    npoints=1000;
    mu=1; timeEnd=100;
    sigma=3*(10**(-4))
    (t,brightness,randNoise)=makeNullData(mu,sigma,npoints,timeEnd)
    f1 = plt.figure(1)
    plt.title('Null data w/o noise')
    plt.ylabel('Brightness')
    plt.xlabel('Phase (hrs)')
    plt.plot(t,brightness,'r--')
    
    f2 = plt.figure(2)
    plt.title('Null data, Gaussian noise')
    plt.ylabel('Brightness')
    plt.xlabel('Phase (hrs)')
    plt.plot(t, brightness+randNoise)

    f3 = plt.figure(3)
    delta=np.linspace(0,2,1000)
    diffLL=computeDeltaLL(delta,t,mu,brightness,randNoise,30,40)
    plt.title('Difference in LL1, LL2 vs. depth of box model')
    plt.ylabel('Delta LL')
    plt.xlabel('box depth')
    plt.plot(delta,diffLL)
    plt.show()
    
      
    

main()
#
#
#ll=0
##LOG LIKELIHDOD
#for i in datapoints:
#    ll += -((i-mu)/(2*(sigma**2)))-(np.log(2*np.pi*(sigma**2))/(2))
#print(ll)
#      
#unit flux- deviation 10-4
#sd 3*10-4
#Fitting a line to the data (is this even relevant? ask)
#ones=np.ones(npoints)
#aMatrix=np.vstack((t, ones)).T
#print(aMatrix)
#m,b = np.linalg.lstsq(aMatrix,brightness,rcond=None)[0]
#print(m,b)



#
#