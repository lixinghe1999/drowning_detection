# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 11:04:08 2020

@author: HeLix
"""

from brping import Ping360
from brping import definitions
import numpy as np
import time
import matplotlib.pyplot as plt
from sonar_display import show_sonar, temporal_info

_firmwareMaxTransmitDuration=500    
_firmwareMinTransmitDuration = 5
_samplePeriodSickDuration=25e-9
_firmwareMaxNumberOfPoints = 1200
_firmwareMinSamplePeriod = 80

def calsampleperiod(distance,number_sample):
    return 2*distance/(number_sample*speed_of_sound*_samplePeriodSickDuration)
def adjustTransmitDuration(distance,sample_period):
    transmit_duration=round(8000*distance/speed_of_sound)
    transmit_duration = max(2.5*sample_period/1000, transmit_duration)
    return max(_firmwareMinTransmitDuration, min(transmitDurationMax(sample_period), transmit_duration))

def transmitDurationMax(sample_period):
    return min(_firmwareMaxTransmitDuration, samplePeriod(sample_period) * 64e6)

def samplePeriod(sample_period):
    return sample_period*_samplePeriodSickDuration

def scan_list(start, stop, step):
    if start>stop:
        angle_list=list(range(start,400,step))+list(range(0,stop,step))
        angle_list+=angle_list[::-1][1:]
    else:
        angle_list=list(range(start,stop,step))
        angle_list+=angle_list[::-1][1:]
    return angle_list

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Control the sonar')
    parser.add_argument('--fake', action="store", required=False, type=bool, default=False,help="We use fake data if it is True")
    parser.add_argument('--real', action="store", required=False, type=bool, default=False,help="We use sonar data if it is True")
    parser.add_argument('--mode', action="store", required=False, type=int, default=0,help="0-scan one sector, 1-scan one direction, 2-auto_transmit(not available now)")

    args = parser.parse_args()

    if args.real:
        device='COM3'
        baudrate=115200
        p = Ping360()
        p.connect_serial(device, baudrate)
        print("Initialized: %s" % p.initialize())
        # setting
        speed_of_sound=1500
        number_sample=300
        frequency=750
        distance=2
        gain_setting=3

        sample_period=calsampleperiod(distance, number_sample)
        sample_period=round(sample_period)
        transmit_duration=adjustTransmitDuration(distance, sample_period)

        print(sample_period,transmit_duration)
        print(p.set_gain_setting(gain_setting))
        print(p.set_transmit_frequency(frequency))
        print(p.set_sample_period(sample_period))
        print(p.set_number_of_samples(number_sample))
        print(p.set_transmit_duration(transmit_duration))

    if args.mode==0:
        # scan sector
        if args.fake:
            number_sample=500

        start_angle=100
        stop_angle=300
        scan_step=1
        repeat=1
        distance=2
        sonar_img=np.zeros((number_sample,int(400/scan_step)))


        for i in range(repeat):
            fig=plt.figure(1)
            local_time=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            fileObject = open("sector_scan/"+local_time+'.txt', 'w')
            for x in scan_list(start_angle,stop_angle,scan_step):
                if args.real:
                    p.transmitAngle(x)
                    new_message=[int(j) for j in p._data]
                if args.fake:
                    # fake data
                    new_message = np.random.random((number_sample))*1

                fileObject.write(str(x)+" ")
                for j in range(len(new_message)):
                    fileObject.write(str(new_message[j])+" ")
                fileObject.write("\n")

                sonar_img[:,int(x/scan_step)]=new_message
            fileObject.close()
            show_sonar(sonar_img,distance,fig)
            plt.savefig("sector_scan/"+local_time+".png",dpi=200,bbox_inches = 'tight')
            plt.show()

    elif args.mode==1:
    # continously scan one direction
        if args.fake:
            number_sample=666

        vertical_span=20
        bins_freq=3
        repeat=10

        point_at=300
        while(1):
            sonar_img=np.zeros((number_sample,repeat))

            fig=plt.figure(1)
            t_start= time.time()
            local_time=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            fileObject = open("one_direction/"+local_time+'.txt', 'w')
            for i in range(repeat):

                #scan adjacent angle
                #scan_angle=point_at+i%3

                scan_angle = point_at
                if args.real:
                    p.transmitAngle(scan_angle)
                    new_message=[int(j) for j in p._data]
                if args.fake:
                    new_message = np.random.random((number_sample))*0.5
                    new_message[30]+=np.sin(2*np.pi*2*(time.time()))
                    new_message[77]+=np.sin(2*np.pi*2*(time.time()))*0.6
                    new_message[175]+=np.sin(2*np.pi*2*(time.time()))*0.5
                    new_message[277]+=np.sin(2*np.pi*2*(time.time()))*0.6
                    plt.pause(0.1)


                fileObject.write(str(scan_angle)+" ")
                for j in range(len(new_message)):
                    fileObject.write(str(new_message[j])+" ")
                fileObject.write("\n")
                sonar_img=np.column_stack((new_message,sonar_img[:,:-1]))

            fileObject.close()
            t_end= time.time()
            T=(t_end-t_start)/repeat

            freq=np.fft.fftfreq(repeat, d=T)
            freq=np.fft.fftshift(freq)
            freq=np.round(freq,1)

            temporal_info(sonar_img, freq, vertical_span, bins_freq, 4)

            fig.suptitle("FPS is "+ str(1/T) +' Hz', fontsize=14)
            plt.savefig("one_direction/"+local_time+".png",dpi=100,bbox_inches = 'tight')
            plt.show()
            break

    elif args.mode==2:
    # turn on auto-scan with 1 grad steps
        start_angle=200
        stop_angle=400
        scan_step=1
        p.control_auto_transmit(start_angle,stop_angle,scan_step,0)

        sonar_img=np.zeros((number_sample,int(400/scan_step)))
        tstart_s = time.time()
        # wait for 400 device_data messages to arrive
        for x in scan_list(start_angle,stop_angle,scan_step):
            p.wait_message([definitions.PING360_DEVICE_DATA])
            new_message=[int(j) for j in p._data]
            sonar_img=update_img(sonar_img, new_message,int(x/scan_step))
            show_sonar(sonar_img,distance,fig)
            tend_s = time.time()

        print("full scan in %dms, %dHz" % (1000*(tend_s - tstart_s), 400/(tend_s - tstart_s)))

        # stop the auto-transmit process
        p.control_motor_off()
        # In[7]

        # turn on auto-transmit with 10 grad steps
        scan_step=10
        p.control_auto_transmit(start_angle,stop_angle,scan_step,0)
        sonar_img=np.zeros((number_sample,int(400/scan_step)))

        tstart_s = time.time()
        # wait for 40 device_data messages to arrive (40 * 10grad steps = 400 grads)
        for x in scan_list(start_angle,stop_angle,scan_step):
            p.wait_message([definitions.PING360_DEVICE_DATA])
            new_message=[int(j) for j in p._data]
            sonar_img=update_img(sonar_img, new_message,int(x/scan_step))
            show_sonar(sonar_img,distance,fig)
            tend_s = time.time()

        print("full scan in %dms, %dHz" % (1000*(tend_s - tstart_s), 400/(tend_s - tstart_s)))

        p.control_reset(0, 0)