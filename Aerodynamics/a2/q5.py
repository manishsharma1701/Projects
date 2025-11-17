# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 22:42:12 2025

@author: Leovo
"""

import numpy as np
import matplotlib.pyplot as plt
import math

alpha=np.array([-18.25869263,-16.1891516,-14.18636996,-12.31710709,
       -10.0472879,-8.244784423,-6.041724618,-3.972183588,
       -1.969401947,0.100139082,2.102920723,4.172461752,
       6.175243394,8.178025035,10.24756606,12.31710709,14.31988873,
       16.1891516,18.3922114,20.46175243])
cm_c4=[0.058477364,0.014214108,-0.011522714,-0.027172962,
        -0.045273945,-0.044925849,-0.041972309,-0.04241534,
        -0.040343173,-0.038258114,-0.037028645,-0.034100889,
        -0.035399509,-0.035855433,-0.032927677, -0.027471832,
        -0.021186182,-0.015769014,-0.038096373,-0.105955134]
alpha_cl=[20.68388, 18.55404, 17.89052, 17.49329, 15.82437, 15.42251, 14.3522, 12.27853,
          10.27126, 8.129622, 6.187562, 4.11191, 2.170186, 0.09522, -1.91336, -4.05564,
          -5.99774, -8.07335, -10.1487, -12.1567, -14.2281, -16.2946, -18.2931, -18.2268]
cl=[1.206309, 1.438486, 1.584895, 1.704786, 1.706963, 1.659952, 1.582435, 1.434122,
    1.272504, 1.08075, 0.862431, 0.64402, 0.439054, 0.244008, 0.035658, -0.17946,
    -0.39778, -0.61619, -0.82459, -1.01291, -1.07777, -1.01845, -0.91565, -0.83355]
plt.plot(alpha,cm_c4,"o-",markersize="5",color="green",label="exp. C_L")
plt.plot(alpha_cl,cl,"s-",markersize="5",color="blue",label="exp. CM_c/4")
plt.xlim(-32,32)
plt.grid("True")

# Expression
p=0.4
m=0.02
theta_p=(np.arccos(1-(2*p)))    

term1 = (m / (np.pi * p**2)) * ((2 * p - 1) * theta_p + np.sin(theta_p))
term2 = (m / (np.pi * (1 - p)**2)) * ((2 * p - 1) * (np.pi - theta_p) - np.sin(theta_p))
A0 = term1 + term2

A1 = (2*m)/(np.pi*p**2) * ((2*p - 1)*np.sin(theta_p) + (1/4)*np.sin(2*theta_p) + (theta_p/2)) \
       - (2*m)/(np.pi*(1 - p)**2) * ((2*p - 1)*np.sin(theta_p) + (1/4)*np.sin(2*theta_p) - (1/2)*(np.pi - theta_p))

A2 = (2*m)/(np.pi*p**2) * ((2*p - 1) * (1/4 * np.sin(2*theta_p) + theta_p/2) + np.sin(theta_p) - (1/3) * np.sin(theta_p)**3) \
       - (2*m)/(np.pi*(1 - p)**2) * ((2*p - 1) * (1/4 * np.sin(2*theta_p) - (np.pi - theta_p)/2) + np.sin(theta_p) - (1/3) * np.sin(theta_p)**3)
print(A0,A1,A2)
Cl_c=np.pi*(A1 - 2*A0) + (2*np.pi*np.radians(alpha))
Cm_c4_c= (-np.pi/4)*(A1-A2)
cmc4=np.ones(len(alpha))*Cm_c4_c
plt.plot(alpha,Cl_c,label="calculated C_L",color="k")
plt.plot(alpha,cmc4,label="Calculated CM_c/4",color="red")
plt.title("Comparison between thin airfoil and experimental value of 2412 airfoil")
plt.xlabel("angle of attack")
plt.ylabel("CM_c/4 and C_L")
plt.legend()
plt.show()
































