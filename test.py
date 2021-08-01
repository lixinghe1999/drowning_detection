import time
from data import *
number_sample = 500
start_angle = 0
stop_angle = 399
distance = 20
fast_scan = 3
slow_scan = 1
threshold = [0, 100, 1.5, 100]
angle = start_angle
angle_former = (start_angle-1)%400
object_former = []
object_record = {}
peaks_record = [[[], []]] * 400
sonar_img = np.zeros((number_sample, 400))
local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
fileObject = open("mode_2/" + local_time + '.txt', 'w')
while(1):
    if (angle > stop_angle):
        fileObject.close()
        angle = angle - stop_angle - 1 + start_angle
        object_record = {}
        peaks_record = [[]] * 400
        local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        fileObject = open("mode_2/" + local_time + '.txt', 'w')
    data = np.random.random((500))
    # record data
    fileObject.write(str(angle) + " ")
    for j in range(len(data)):
        fileObject.write(str(data[j]) + " ")
    fileObject.write("\n")
    data[:round(number_sample/distance)] = 0

    len_sample = distance / number_sample
    data_filter = smooth(data, len_sample, 0)
    local_var = smooth(abs(data - data_filter), len_sample, 1)
    peaks, dict = detect(data_filter, len_sample, local_var)
    new_object, overlap = update_record(peaks_record, object_record, dict, angle, angle_former, len_sample)
    sonar_img[:, angle] = data
    rmax = 0
    angle_add = fast_scan
    for o in object_former:
        if o[1] not in overlap:
            if filter(object_former[o], threshold):
                if object_former[o][4] > rmax:
                    rmax = object_former[o][4]
                    r = object_former[o]
        else:
            angle_add = slow_scan
    if rmax != 0:
        print(r)
    angle_former = angle
    object_former = new_object
    for i in range(1, angle_add):
        sonar_img[:, (angle + i) % 400] = data
    angle = angle + angle_add