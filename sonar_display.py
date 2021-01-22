# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 11:21:55 2020

@author: HeLix
"""
import numpy as np
import matplotlib.pyplot as plt

def get_data(decoded_message):
    data=[int(item) for item in decoded_message.data]
    angle=decoded_message.angle
    return data, angle
def show_sonar(data2D,distance,fig):
    num_samples, num_degree=np.shape(data2D)
    r=np.linspace(0,distance,num_samples)
    theta=np.linspace(0,2*np.pi,num_degree)
    theta,r=np.meshgrid(theta,r)
    X=r*np.cos(theta)
    Y=r*np.sin(theta)
    ax=fig.add_subplot(111)
    ax.pcolormesh(X,Y,data2D, cmap='Greys',antialiased=True)
def temporal_info(data2D, freq, vertical_span=10, bins_freq=3, num_figure=1, mode=0):
    # argument
    # data2D： (num_samples, num_time) raw data from sonar
    # freq: frequency vector
    # vertical_span, bins_freq: only for display
    # num_figure: the object we need to show, include some strong noise
    # mode: show frequency or shape
    d1, d2= np.shape(data2D)    
    fft_data=abs(np.fft.fft(data2D,axis=1))
    sum_data=np.sum(data2D,axis=1)
    for i in range(num_figure):
        highlight=np.argmax(sum_data)
        if (highlight-vertical_span)<0:
            vertical_span = highlight
        elif (highlight+vertical_span)>=d1:
            vertical_span = d1 - highlight -1
        sum_data[highlight-vertical_span:highlight+vertical_span]=0
        if mode==0:
            crop_fft_data=np.fft.fftshift(fft_data[highlight-vertical_span:highlight+vertical_span,:],axes=1)
            plt.subplot(num_figure,2,2*i+1)
            plt.imshow(data2D[highlight-vertical_span:highlight+vertical_span,:],cmap='gray',aspect=d2/(2*vertical_span))
            plt.yticks([2*vertical_span-1,vertical_span,0], [highlight+vertical_span-1,highlight,highlight-vertical_span])
            plt.subplot(num_figure,2,2*i+2)
            plt.imshow(crop_fft_data,cmap='gray',aspect=d2/(2*vertical_span))
            plt.yticks([2*vertical_span-1,vertical_span,0], [highlight+vertical_span-1,highlight,highlight-vertical_span])
            plt.xticks(np.arange(d2)[::int(d2/bins_freq)], freq[::int(d2/bins_freq)])
            print(np.shape(data2D),highlight,vertical_span)
        else:
            plt.subplot(num_figure,1,i+1)
            for j in range(d2):
                plt.plot(range(highlight-vertical_span,highlight+vertical_span),data2D[highlight-vertical_span:highlight+vertical_span,j])
    


if __name__=='__main__':
    num_sample=666
    num_t=10
    f=2
    T=1/10
    #1/T> 2*2*pi*f
    T_max=1/(2*f)
    print(T_max, T)
    t=num_t * T
    fig=plt.figure(1)
    freq=np.fft.fftfreq(num_t, d=T)
    freq=np.fft.fftshift(freq)
    data2D=np.random.random((num_sample,num_t))*0.5
    data2D[128,:]+=np.sin(2*np.pi*f*np.linspace(0,t,num_t))
    
    vertical_span=10
    bins_freq=3
    temporal_info(data2D, freq, vertical_span, bins_freq, 2,0)
    plt.show()

'''
num_frames=20
num_samples=500
test=np.random.random((num_samples,num_frames))*255

for i in range(20):
    new_message=np.ones((num_samples,1))*255
    test=update_img(test, new_message)
    plt.pause(0.1)
'''
