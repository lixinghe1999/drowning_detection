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
from sonar_display import show_sonar
from data import *
min_human=0.4
max_human=2
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

def rescan(start, end, k, images):
    for i in range(k):
        image = np.zeros((500, end-start+1))
        for j in range(start, end+1):
            p.control_transducer(
                0,  # reserved
                p._gain_setting,
                i,
                p._transmit_duration,
                p._sample_period,
                p._transmit_frequency,
                p._number_of_samples,
                1,
                0
            )
            p.wait_message([definitions.PING360_DEVICE_DATA], 0.5)
            data = [int(j) for j in p._data]
            len_sample = distance / len(data)
            data_filter = smooth(data, len_sample, 0)
            image[:, j-start] = data_filter
        images = np.concatenate((images, image[:,:, np.newaxis]), axis=2)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Control the sonar')
    parser.add_argument('--data', action="store", required=False, type=int, default=0, help="We use fake data or sonar data")
    parser.add_argument('--mode', action="store", required=False, type=int, default=0, help="0-scan one sector, 1-scan one direction, 2-auto_transmit(not available now)")
    args = parser.parse_args()

    if args.data == 1:
        device='COM4'
        baudrate=115200
        p = Ping360()
        p.connect_serial(device, baudrate)
        p.initialize()
        # setting
        speed_of_sound = 1500
        number_sample = 250
        frequency = 750
        distance = 10
        gain_setting = 0

        sample_period = calsampleperiod(distance, number_sample)
        sample_period = round(sample_period)
        transmit_duration = adjustTransmitDuration(distance, sample_period)

        print(sample_period, transmit_duration)
        p.set_gain_setting(gain_setting)
        p.set_transmit_frequency(frequency)
        p.set_sample_period(sample_period)
        p.set_number_of_samples(number_sample)
        p.set_transmit_duration(transmit_duration)

    if args.mode == 0:
        # scan sector
        if args.data == 0:
            number_sample=500
            distance=10
        # adjust the start and end angle
        start_angle=170
        stop_angle=280
        scan_step=2
        repeat=1
        sonar_img=np.zeros((number_sample,int(400/scan_step)))
        for i in range(repeat):
            fig=plt.figure(1)
            local_time=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            fileObject = open("sector_scan/"+local_time+'.txt', 'w')
            for x in range(start_angle,stop_angle,scan_step):
                if args.data:
                    p.control_transducer(
                        0,  # reserved
                        p._gain_setting,
                        x,
                        p._transmit_duration,
                        p._sample_period,
                        p._transmit_frequency,
                        p._number_of_samples,
                        1,
                        0
                    )
                    p.wait_message([definitions.PING360_DEVICE_DATA], 0.5)
                    new_message = [int(j) for j in p._data]
                    #new_message = np.ones(number_sample)
                else:
                    # fake data
                    new_message = np.random.random((number_sample))*1
                fileObject.write(str(x)+" ")
                for j in range(len(new_message)):
                    fileObject.write(str(new_message[j])+" ")
                fileObject.write("\n")
                if len(new_message)>0:
                    sonar_img[:,int(x/scan_step)]=new_message
            fileObject.close()
            show_sonar(sonar_img, distance)
            plt.savefig("sector_scan/"+local_time+".png",dpi=200,bbox_inches = 'tight')
            plt.show()
    elif args.mode == 1:
    # continously scan smaller sector
        if args.data == 0:
            number_sample=666
        repeat = 30
        start_angle = 163
        end_angle = 173
        for r in range(repeat):
            local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            fileObject = open("one_direction/" + local_time + '.txt', 'w')
            t_start = time.time()
            for i in range(start_angle, end_angle, 1):
                if args.data:
                    p.control_transducer(
                        0,  # reserved
                        p._gain_setting,
                        i,
                        p._transmit_duration,
                        p._sample_period,
                        p._transmit_frequency,
                        p._number_of_samples,
                        1,
                        0
                    )
                    p.wait_message([definitions.PING360_DEVICE_DATA], 0.5)
                    new_message=[int(j) for j in p._data]
                else:
                    new_message = np.random.random((number_sample))*0.5
                    new_message[30]+=np.sin(2*np.pi*2*time.time())+1
                    new_message[77]+=(np.sin(2*np.pi*2*time.time())+1)*0.6
                    new_message[175]+=(np.sin(2*np.pi*2*time.time())+1)*0.5
                    new_message[277]+=(np.sin(2*np.pi*2*time.time())+1)*0.6
                    plt.pause(0.1)
                fileObject.write(str(i)+" ")
                for j in range(len(new_message)):
                    fileObject.write(str(new_message[j])+" ")
                fileObject.write("\n")
            fileObject.close()
            print((end_angle-start_angle)/(time.time()-t_start))
    else:
    # real deployment
        i = 0
        angle_former = 0
        object_record = {}
        peaks_record = [[]] * 400
        sonar_img = np.zeros((number_sample, 400))
        while(1):
            if (i>99):
                i = 0
                object_record = {}
                peaks_record = [[]] * 400
            p.control_transducer(
                0,  # reserved
                p._gain_setting,
                i,
                p._transmit_duration,
                p._sample_period,
                p._transmit_frequency,
                p._number_of_samples,
                1,
                0
            )
            p.wait_message([definitions.PING360_DEVICE_DATA], 0.5)
            data=[int(j) for j in p._data]
            len_sample = distance / len(data)
            data_filter = smooth(data, len_sample, 0)
            local_var = smooth(abs(data - data_filter), len_sample, 1)
            peaks, dict = detect(data_filter, len_sample, local_var)
            new_object = update_record(peaks_record, object_record, dict, angle, angle_former, len_sample)
            sonar_img[:, i] = data_filter
            if len(new_object) == 0:
                i = i + 4
                continue
            else:
                for k in range(len(new_object)):
                    # if we find object
                    if new_object[k][0] > 0.5:
                        images = sonar_img[:, new_object[k][1]: new_object[k][2], np.newaxis]
                        rescan(new_object[k][1], new_object[k][2], 2, images)
            i = i + 2

