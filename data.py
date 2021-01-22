'''
analyze the response data
'''
import numpy as np
import matplotlib.pyplot as plt

min_human=0.25
max_human=4

def filter(data, scan_range):
    len_sample=scan_range/len(data)
    len_filter=round(min_human/len_sample)
    return np.convolve(data, np.ones((len_filter,)) / len_filter, mode="same")

def detect(data, scan_range):
    len_sample = scan_range / len(data)
    len_block = round(max_human / len_sample)
    var = np.empty(len(data))
    mea = np.empty(len(data))
    print(len_block)
    for i in range(0,len(data),round(len_block)):
        var[i:i+len_block]=np.var(data[i:i+len_block])
        mea[i:i+len_block]=np.mean(data[i:i+len_block])
    return np.logical_or(data < mea-2*var, data > mea+2*var)


def adjust_detect(index):
    a=0
    peak=[]
    for i in range(len(index)):
        if index[i] and a==0:
            a=1
            peak.append(i)
        elif a==1 and (not index[i]):
            a=0
            peak.append(i)
    return peak



if __name__ == '__main__':
    data=np.random.random((500))
    data[30:50]+=8
    data_filter=filter(data,25)
    index=detect(data_filter,25)
    peak=adjust_detect(index)
    print(peak)
    plt.plot(data[index])
    #plt.show()
