'''
analyze the response data
'''
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
import glob
from imageread import sonar2pool
from sonar_display import show_sonar
min_human=0.4
max_human=2

def smooth(data, len_sample, k):
    if k == 0:
        len_filter=round(min_human/len_sample)
    else:
        len_filter=round(max_human/len_sample)
    return np.convolve(data, np.ones(len_filter)/len_filter, mode="same")
def detect(data, len_sample, threshold):
    max_peak = round(max_human / len_sample)
    min_peak = round(min_human / len_sample)
    peaks, dict = find_peaks(data,distance = min_peak, width = [min_peak,max_peak],prominence=2.5*threshold)
    return peaks, dict
def cal_size(widths, len_sample):
    return [i*len_sample for i in widths]
def compensate(data):
    # compensate the further and closer response
    if len(np.shape(data))==2:
        d1, d2 = np.shape(data)
        weight = np.arange(0, 1, 1 / d1)
        weights = np.tile(weight[:, np.newaxis], (1, d2))
    else:
        d1 = len(data)
        weights = np.arange(0, 1, 1 / d1)
    return data*weights
def readline(line):
    line = line.split()
    line = list(map(int, line))
    angle = line[0]
    data = line[1:]
    return angle, data
def update_record(peaks_record, object_record, dict, angle, a, len_sample):
    new_object = {}
    overlap = {}
    if len(dict['left_ips'])>0:
        peaks_record[angle] = np.array([dict['left_ips'], dict['right_ips']])
        J = len(peaks_record[angle][1])
        K = len(peaks_record[a][1])
        if len(peaks_record[a]) != 0 and len(peaks_record[angle]) != 0:
            ratio_mat = np.zeros((J, K))
            for j in range(J):
                 for k in range(K):
                    ratio = (min(peaks_record[a][1][k], peaks_record[angle][1][j]) - max(peaks_record[a][0][k], peaks_record[angle][0][j]))/\
                            (max(peaks_record[a][1][k], peaks_record[angle][1][j]) - min(peaks_record[a][0][k], peaks_record[angle][0][j]))
                    ratio_mat[j, k] = ratio
            for j in range(J):
                if K > 0:
                    if max(ratio_mat[j, :]) > 0.5:
                        kmax = np.argmax(ratio_mat[j, :])
                        if kmax not in overlap:
                            overlap[kmax] = [j]
                        else:
                            overlap[kmax].append(j)
                    else:
                        object_record[angle, j] = [len_sample ** 2 * (peaks_record[angle][1][j] ** 2 - peaks_record[angle][0][j] ** 2) * np.pi / 200, \
                                                   angle, angle, len_sample * peaks_record[angle][0][j],len_sample * peaks_record[angle][1][j], dict['prominences'][j]]
                        new_object[angle, j] = object_record[angle, j]
                else:
                    object_record[angle, j] = [len_sample ** 2 * (peaks_record[angle][1][j] ** 2 - peaks_record[angle][0][j] ** 2) * np.pi / 200, \
                                               angle, angle, len_sample * peaks_record[angle][0][j],len_sample * peaks_record[angle][1][j], dict['prominences'][j]]
                    new_object[angle, j] = object_record[angle, j]
            for k in overlap.keys():
                js = overlap[k]
                for j in js:
                    object_record[angle, j] = object_record[a, k]
                    object_record[angle, j][0] += len_sample**2 * (peaks_record[angle][1][j]**2 - peaks_record[angle][0][j]**2) * np.pi / 200
                    object_record[angle, j][1] = min(object_record[angle, j][1], angle)
                    object_record[angle, j][2] = max(object_record[angle, j][2], angle)
                    object_record[angle, j][3] = min(object_record[angle, j][3], len_sample*peaks_record[angle][0][j])
                    object_record[angle, j][4] = max(object_record[angle, j][4], len_sample*peaks_record[angle][1][j])
                    object_record[angle, j][5] = dict['prominences'][j] * 1/((object_record[angle, j][2] - object_record[angle, j][1])+1) + \
                                                 object_record[angle, j][5] * (1- 1/((object_record[angle, j][2] - object_record[angle, j][1])+1))
                    new_object[angle, j] = object_record[angle, j]
                object_record.pop((a,k))
        elif len(peaks_record[a]) == 0 and len(peaks_record[angle]) != 0:
            for j in range(len(peaks_record[angle][1])):
                object_record[angle, j] = [len_sample**2 * (peaks_record[angle][1][j]**2 - peaks_record[angle][0][j]**2) * np.pi / 200, \
                                        angle, angle, len_sample * peaks_record[angle][0][j], len_sample * peaks_record[angle][1][j], dict['prominences'][j]]
                new_object.append(object_record[angle, j])
    return new_object, overlap
def compare(g, results, threshold):
    if len(results)==0:
        return 100, []
    else:
        c = 100 * np.ones((len(results)))
        keys = list(results.keys())
        for i in range(len(keys)):
            result = results[keys[i]]
            if result[0] <= threshold:
                continue
            c[i] = abs((g[0] - (result[4] + result[3])/2))/20 + abs((g[1] - (result[2] + result[1])/2))/100
        return min(c), results[keys[np.argmin(c)]]
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='analyse experiment data')
    parser.add_argument('--exp', action="store", required=False, type=int, default=4, help="We have collected multiple dataset")
    args = parser.parse_args()
    if args.exp == 1:
        # for the first experiment
        ref_20 = "first_dataset/20_big/20m_ref.txt"
        ref_10 = "first_dataset/10_big/10m_ref.txt"
        dir_20 = "first_dataset/20_small/"
        dir_10 = "first_dataset/10_small/"
        ref = ref_10
        rootdir = dir_10
        f = open(ref,"r")
        num_sample = 500
        scan_range = 10
        sonar_image_ref = np.zeros((num_sample, 400))
        lines = f.readlines()
        for line in lines:
            angle, data = readline(line)
            if len(data) == num_sample:
                sonar_image_ref[:,angle] = data
        f.close()

        files = glob.glob(rootdir+'*.txt')
        a = []
        correct = 0
        whole = 0
        i = 0
        for file in files:
            f = open(file, "r")
            lines = f.readlines()
            #lines = lines[::2]
            data_buffer = np.zeros((num_sample, len(lines)))
            angle_former = 0
            object_record = {}
            peaks_record = [[]] * 400
            j = 0
            plt.figure(1)
            for line in lines:
                if j>=1:
                    angle_former = angle
                angle, data = readline(line)
                if angle < angle_former:
                    whole += 1
                    for k in list(object_record.keys()):
                        if object_record[k][0] <= 0.4:
                            del object_record[k]
                        elif object_record[k][3]>=6 and object_record[k][4]<=8:
                            correct+=1
                            break
                    print(object_record)
                    object_record = {}
                    angle_former = 0
                    peaks_record = [[]] * 400
                data = abs(data-sonar_image_ref[:, angle])
                # whether compensate attenuation
                # data = compensate(data)
                len_sample = scan_range / len(data)
                data_filter = smooth(data, len_sample, 0)
                local_var = smooth(abs(data - data_filter), len_sample, 1)
                peaks, dict = detect(data_filter, len_sample, local_var)
                update_record(peaks_record, object_record, dict, angle, angle_former, len_sample)
                #plt.plot(peaks, data_filter[peaks], linestyle='None', marker='o', markersize=10)
                #plt.plot(data_filter)
                data_buffer[:, j] = data_filter
                j = j + 1
                # 233 for 20 meters, 240 for 10 meters
                if angle == 240:
                    a.append(data_filter)
            plt.xlabel('distance')
            plt.ylabel('response')
            plt.title('scan result from multiple beams')
            f.close()
            #plt.pause(0.01)
            #plt.clf()
            #plt.figure(2)
            freq = np.fft.fftfreq(len(lines), d=1/11)
            freq = np.fft.fftshift(freq)
            freq = np.round(freq, 1)
            #temporal_info(data_buffer, freq, vertical_s=round(max_human/len_sample/2), bins_freq=3, num_figure=3, mode=0)
            #plt.pause(0.01)
            #plt.clf()
        plt.figure(3)
        plt.plot(np.sqrt(np.var(a,0))/np.mean(a,0))
        plt.show()
        print(correct/whole)
    elif args.exp == 2:
        reference = "second_dataset/reference.txt"
        f = open(reference, "r")
        num_sample = 500
        scan_range = 20
        sonar_image_ref = np.zeros((num_sample, 400))
        detection = True
        lines = f.readlines()
        for line in lines:
            angle, data = readline(line)
            if len(data) == num_sample:
                sonar_image_ref[:, angle] = data
        f.close()
        color = {'5/':'b', '10/': 'r', '15/':'y'}
        o = 0
        for dis in ['5/','10/','15/']:
            if dis == '5/':
                if detection:
                    directories = ['static1', 'static3', 'Noaction1A', 'Noaction2A', 'Noaction2B','action1A', 'action1B', 'action2A']
                else:
                    directories = ['Noaction1A', 'Noaction2A', 'Noaction2B', 'action1A', 'action1B', 'action2A']
                ground_truths = {'static1': [(5.5,170),(4.2, 203)],'static3':[(5,168),(4.5,204)],'Noaction1A':[(5.2,160)],'Noaction2A':[(5.2,157)],'Noaction2B':[(5,200)],'action1A':
                                 [(5.2,160)],'action1B':[(5, 195)],'action2A':[(6, 165)]}
            elif dis == '10/':
                if detection:
                    directories = ['static1', 'static3', 'Noaction1B', 'Noaction2A', 'Noaction2B', 'action1A', 'action1B', 'action2A', 'action2B']
                else:
                    directories = ['Noaction1B', 'Noaction2A', 'Noaction2B', 'action1A', 'action1B', 'action2A', 'action2B']
                ground_truths = {'static1': [(10, 178), (10, 196)], 'static3': [(9.5, 175), (9.3, 195)],'Noaction1B': [(8.5, 200)], 'Noaction2A': [(7.5, 175)],
                                 'Noaction2B': [(8, 196)], 'action1A':[(7.8, 175)], 'action1B': [(7.8, 192)], 'action2A': [(7, 170)], 'action2B':[(7.2, 190)]}
            elif dis == '15/':
                if detection:
                    directories = ['static1', 'static3', 'Noaction1A', 'action1A', 'action2A']
                else:
                    directories = ['Noaction1A', 'action1A', 'action2A']
                ground_truths = {'static1': [(16, 185)], 'static3': [(16, 185)],'Noaction1A': [(16.7, 185)], 'action1A': [(17, 184)], 'action2A': [(17, 186)]}
            directories = directories
            for d in directories:
                path = 'second_dataset/' + dis + d +'/*.txt'
                files = glob.glob(path)
                ground_truth = ground_truths[d]
                c_threshold = 0.2
                correct = 0
                size = []
                intensity = []
                for file in files:
                    f = open(file, 'r')
                    lines = f.readlines()
                    sonar_img = np.zeros((num_sample, len(lines)))
                    angle_former = 0
                    object_record = {}
                    peaks_record = [[[],[]]] * 400
                    for i in range(len(lines)):
                        if i >= 1:
                            angle_former = angle
                        angle, data = readline(lines[i])
                        data = abs(data - sonar_image_ref[:, angle])
                        len_sample = scan_range / len(data)
                        data_filter = smooth(data, len_sample, 0)
                        local_var = smooth(abs(data - data_filter), len_sample, 1)
                        peaks, dict = detect(data_filter, len_sample, local_var)
                        new_object, overlap = update_record(peaks_record, object_record, dict, angle, angle_former, len_sample)
                        sonar_img[:, i] = data_filter
                        if i == 0:
                            first_angle = angle
                        elif i == len(lines)-1:
                            last_angle = angle
                    f.close()
                    for g in ground_truth:
                        c, r= compare(g, object_record, 0.5)
                        if c <= c_threshold:
                            correct = correct + 1
                            crop_image = sonar_img[int(r[3]/len_sample):int(r[4]/len_sample)+1,r[1]-first_angle:r[2]-first_angle+1]
                            np.save('second_dataset/' + 'images/'  + d + '-' + str(o) + '.npy', crop_image)
                            o = o + 1
                            # plt.scatter(int(r[4]/len_sample)-int(r[3]/len_sample), r[2]-r[1], c = color[dis])
                            # size.append(r[0])
                            # intensity.append(r[5])
                    # plt.imshow(sonar_img, cmap='gray',aspect=len(lines)/num_sample)
                    # plt.yticks([0,49,99,149,199,249,299,349,399,449,499],[0,2,4,6,8,10,12,14,16,18,20])
                    # plt.xticks(np.arange(len(lines))[::int(len(lines)/4)], np.arange(first_angle, last_angle+1)[::int(len(lines)/4)])
                    # plt.show()
                print(dis[:-1], d, correct/len(files)/len(ground_truth))
    elif args.exp == 3:
        n = 0
        for sonar in ["sonar_1/", "sonar_2/"]:
            reference = "third_dataset/" + sonar + "reference.txt"
            f = open(reference, "r")
            num_sample = 500
            scan_range = 20
            sonar_image_ref = np.zeros((num_sample, 400))
            lines = f.readlines()
            for line in lines:
                angle, data = readline(line)
                if len(data) == num_sample:
                    sonar_image_ref[:, angle] = data
            f.close()
            directories = ['Noaction1', 'Noaction2', 'action1', 'action2']
            ground_truths = {"sonar_1/":{'Noaction1':[(6,167),(3.5, 154)], 'Noaction2':[(6,164),(3.8,155)], 'action1':[(6,165),(4,158)], 'action2':[(6.6,165),(4,155)],},
                             "sonar_2/":{'Noaction1':[(3.8,232),(5.2,262)], 'Noaction2':[(3,230),(5,260)], 'action1':[(3,230),(5,260)], 'action2':[(3,220),(4.5,260)]}}
            for d in directories:
                path = 'third_dataset/' + sonar + d
                files = os.listdir(path)
                ground_truth = ground_truths[sonar][d]
                c_threshold = 0.2
                correct = 0
                size = []
                intensity = []
                for file in files:
                    #print(file)
                    f = open(path+'/'+file, 'r')
                    lines = f.readlines()
                    data_buffer = np.zeros((num_sample, len(lines)))
                    sonar_img = np.zeros((num_sample, 400))
                    angle_former = 0
                    object_former = {}
                    object_record = {}
                    peaks_record = [[[],[]]] * 400
                    for i in range(len(lines)):
                        if i == 0:
                            first_angle = angle
                        elif i == len(lines) - 1:
                            last_angle = angle
                        if i >= 1:
                            angle_former = angle
                        angle, data = readline(lines[i])
                        if len(data) == 0:
                            continue
                        data = abs(data - sonar_image_ref[:, angle])
                        len_sample = scan_range / len(data)
                        data_filter = smooth(data, len_sample, 0)
                        local_var = smooth(abs(data - data_filter), len_sample, 1)
                        peaks, dict = detect(data_filter, len_sample, local_var)
                        new_object, overlap = update_record(peaks_record, object_record, dict, angle, angle_former, len_sample)
                        data_buffer[:, i] = data_filter
                        sonar_img[:, angle] = data_filter
                        object_former = new_object
                    f.close()
                    show_sonar(sonar_img, 20)
                    plt.show()
                    for g in ground_truth:
                        c, r = compare(g, object_record, 0.4)
                        if c <= c_threshold:
                            correct = correct + 1
                            crop_image = sonar_img[int(r[3] / len_sample):(int(r[4] / len_sample) + 1), r[1]:r[2]+1]
                            np.save('third_dataset/' + sonar + 'images/' + d + '-' + str(n) + '.npy', crop_image)
                            n = n + 1
                            # plt.scatter(int(r[4]/len_sample)-int(r[3]/len_sample), r[2]-r[1])
                        # plt.imshow(sonar_img, cmap='gray', aspect=len(lines) / num_sample)
                        # plt.yticks([0, 49, 99, 149, 199, 249, 299, 349, 399, 449, 499],
                        #            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
                        # plt.xticks(np.arange(len(lines))[::int(len(lines) / 4)],
                        #            np.arange(first_angle, last_angle + 1)[::int(len(lines) / 4)])
                        # plt.show()

                print(sonar + d, correct / len(files))
    elif args.exp == 4:
        n = 0
        for sonar in ["sonar_1/", "sonar_2/"]:
            reference = "third_dataset/" + sonar + "reference.txt"
            f = open(reference, "r")
            num_sample = 500
            scan_range = 20
            sonar_image_ref = np.zeros((num_sample, 400))
            lines = f.readlines()
            for line in lines:
                angle, data = readline(line)
                if len(data) == num_sample:
                    sonar_image_ref[:, angle] = data
            f.close()
            directories = ['swim', 'walk']
            ground_truths = {"sonar_1/": {'swim': [(0, 0), (0, 0)], 'walk': [(0, 0), (0, 0)]},
                             "sonar_2/": {'swim': [(0, 0), (0, 0)], 'walk': [(0, 0), (0, 0)]}}
            save_path = "third_dataset/" + sonar + 'result.txt'
            f_save = open(save_path, 'w')
            for d in directories:
                path = 'third_dataset/' + sonar + d
                files = os.listdir(path)
                ground_truth = ground_truths[sonar][d]
                c_threshold = 0.2
                correct = 0
                size = []
                intensity = []
                for file in files:
                    f = open(path+'/'+file, 'r')
                    f_save.write(file+' ')
                    lines = f.readlines()
                    data_buffer = np.zeros((num_sample, len(lines)))
                    sonar_img = np.zeros((num_sample, 400))
                    angle_former = 0
                    object_former = {}
                    object_record = {}
                    peaks_record = [[[], []]] * 400
                    for i in range(len(lines)):
                        if i == 0:
                            first_angle = angle
                        elif i == len(lines) - 1:
                            last_angle = angle
                        if i >= 1:
                            angle_former = angle
                        angle, data = readline(lines[i])
                        if len(data) == 0:
                            continue
                        data = abs(data - sonar_image_ref[:, angle])
                        len_sample = scan_range / len(data)
                        data_filter = smooth(data, len_sample, 0)
                        local_var = smooth(abs(data - data_filter), len_sample, 1)
                        peaks, dict = detect(data_filter, len_sample, local_var)
                        new_object, overlap = update_record(peaks_record, object_record, dict, angle, angle_former,
                                                            len_sample)
                        data_buffer[:, i] = data_filter
                        sonar_img[:, angle] = data_filter
                        # for continuously scan
                        for o in object_former:
                            if o[1] not in overlap:
                                if object_former[o][0] > 0.5:
                                    if sonar == 'sonar_1/':
                                        xy = sonar2pool(1, (object_record[o][1] + object_record[o][2]) / 2,(object_record[o][3] + object_record[o][4]) / 2)
                                    else:
                                        xy = sonar2pool(2, (object_record[o][1] + object_record[o][2]) / 2,(object_record[o][3] + object_record[o][4]) / 2)
                                    if xy[0]<8 and xy[0]>0 and xy[1]>0 and xy[1]<6:
                                        f_save.write(str(xy[0]) + ' ' + str(xy[1]) + ' ')
                        object_former = new_object
                    f.close()
                    f_save.write('\n')
                    # show_sonar(sonar_img, 20)
                    # plt.show()
                    for g in ground_truth:
                        c, r = compare(g, object_record, 0.4)
                        if c <= c_threshold:
                            correct = correct + 1
                            # crop_image = sonar_img[int(r[3] / len_sample):(int(r[4] / len_sample) + 1), r[1]:r[2] + 1]
                            # np.save('third_dataset/' + sonar + 'images/' + d + '-' + str(n) + '.npy', crop_image)
                            # n = n + 1
                print(sonar + d, correct / len(files))
            f_save.close()






