# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:29:45 2020

@author: HeLix
"""
import numpy as np
import matplotlib.pyplot as plt

f=750000
t=np.linspace(0,1,10000)
offset=2
y=offset+np.sin(2*np.pi*f*t)  

a=1
if a==0:
    pass
# add another frequency
else:
    f1=1000000
    y=y+np.sin(2*np.pi*f1*t)*0.1
y_echo=np.pad(y,(1000,10000),'constant',constant_values=(0,0))

# add noise
SNR=20
noise = np.random.randn(y_echo.shape[0]) 	
noise = noise-np.mean(noise)
signal_power = np.linalg.norm( y_echo )**2 / y_echo.size	
noise_variance = signal_power/np.power(10,(SNR/10))        
noise = (np.sqrt(noise_variance) / np.std(noise) )*noise
y_echo_noise=y_echo+noise
    
# compute cross-correlation and quantization
cross_correlate=(np.correlate(y_echo_noise,y))
step=np.max(cross_correlate)/255
q_corr=np.round(cross_correlate/step)


# visualize
plt.subplot(311)
plt.plot(y)
plt.subplot(312)
plt.plot(y_echo_noise)
plt.subplot(313)
plt.plot(q_corr)