'''
analyze the response data
'''
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

min_human=0.5
max_human=2.5

def filter(data, len_sample, k):
    if k == 0:
        len_filter=round(min_human/len_sample)
    else:
        len_filter=round(max_human/len_sample)
    return np.convolve(data, np.ones(len_filter)/len_filter, mode="same")
def detect(data, len_sample, threshold):
    max_peak = round(max_human / len_sample)
    min_peak = round(min_human / len_sample)
    peaks, dict = find_peaks(data,distance = min_peak, width = [min_peak,max_peak],prominence=3*threshold)
    return peaks, dict
def cal_size(widths, len_sample):
    return [i*len_sample for i in widths]

if __name__ == '__main__':
    data=np.random.random((500))*0.5
    data[200:500]+=2
    data[30:50]+=1
    data[350:372]+=1.5
    scan_range = 25
    len_sample = scan_range / len(data)
    data_filter=filter(data, len_sample, 0)
    local_var=filter(abs(data-data_filter), len_sample, 1)
    peaks, dict=detect(data_filter,len_sample,local_var)
    print(peaks, dict['widths'])
    plt.plot(data_filter)
    plt.show()
