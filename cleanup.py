# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:44:40 2020

@author: anura
"""
# =============================================================================
# stuff to clean up and implement in main function
# 
# =============================================================================
# =============================================================================
#    here's where i sweep through four different random start points. just troubleshooting features
# =============================================================================
#    m=2
#    n=2
#     diffLL1=np.zeros(10)
#
#    [fig,a]=makeSubplotsWithNullData(m,n,'time','brightness','Depth $\propto$ mean of points inside box',t,brightness,sigmas)
#    
#    tStartPoints=np.random.randint(1496, size=4)
#    depth=sigma
#    
#    fillSubplots(fig,a,m,n,t,brightness,sigmas,depth,tStartPoint,duration,mu):



  
#    a[0][0].errorbar(t,brightness, yerr=sigmas,fmt='ko',markersize=0.7,ecolor='blue')
#    depth=1-np.mean(brightness[tStartPoints[0]:tStartPoints[0]+duration])
#    print(depth)
#
#    a[0][0].plot(boxModel(t,mu,tStartPoints[0],tStartPoints[0]+duration,depth),'y')
#    diffLL1[0]=computeDeltaLL(depth,mu,brightness,sigmas,tStartPoints[0],duration)
#
#    a[0][1].errorbar(t,brightness, yerr=sigmas,fmt='ko',markersize=0.7,ecolor='blue')
#    depth=1-np.mean(brightness[tStartPoints[1]:tStartPoints[1]+duration])
#    print(depth)
#
#    a[0][1].plot(boxModel(t,mu,tStartPoints[1],tStartPoints[1]+duration,depth),'y')
#    diffLL1[1]=computeDeltaLL(depth,mu,brightness,sigmas,tStartPoints[1],duration)
#
#    a[1][0].errorbar(t,brightness, yerr=sigmas,fmt='ko',markersize=0.7,ecolor='blue')
#    depth=1-np.mean(brightness[tStartPoints[2]:tStartPoints[2]+duration])
#    print(depth)
#
#    a[1][0].plot(boxModel(t,mu,tStartPoints[2],tStartPoints[2]+duration,depth),'y')
#    diffLL1[2]=computeDeltaLL(depth,mu,brightness,sigmas,tStartPoints[2],duration)
#
#    a[1][1].errorbar(t,brightness, yerr=sigmas,fmt='ko',markersize=0.7,ecolor='blue')
#    depth=1-np.mean(brightness[tStartPoints[3]:tStartPoints[3]+duration])
#    print(depth)
#
#    a[1][1].plot(boxModel(t,mu,tStartPoints[3],tStartPoints[3]+duration,depth),'y')
#    diffLL1[3]=computeDeltaLL(depth,mu,brightness,sigmas,tStartPoints[3],duration)
#    print(diffLL1)
#
#DLL<1 for depth mean
    #start pos, depth, deltaLL title info duration
    #
#optimal