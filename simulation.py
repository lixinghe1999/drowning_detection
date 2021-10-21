import numpy as np
import random
import matplotlib.pyplot as plt
import math
import matplotlib.pyplot as plt

random_move = 0.05
span = 40/180 * np.pi
max_time = 1000
min_range = 0.3
max_range = 10

def clock_wise_angle(vector_1, vector_2):
    x1, y1 = vector_1
    x2, y2 = vector_2
    dot = x1 * x2 + y1 * y2
    det = x1 * y2 - y1 * x2
    theta = np.arctan2(det, dot)
    return theta
def swimmers_move(swimmers_loc, swimmers_speed, pool_size):
    for i in range(len(swimmers_loc)):
        if edge(swimmers_loc[i], pool_size):
            if random.random() > 0.5:
                swimmers_speed[i] = [random.uniform(-random_move, random_move), random.uniform(-random_move, random_move)]
            else:
                if swimmers_loc[i][1] > (pool_size[1]/2):
                    swimmers_speed[i] = [random.uniform(-random_move, random_move), -(random.uniform(-random_move, random_move) + 1)]
                else:
                    swimmers_speed[i] = [random.uniform(-random_move, random_move), (random.uniform(-random_move, random_move) + 1)]
        else:
            swimmers_speed[i] = swimmers_speed[i] + [random.uniform(-random_move, random_move), random.uniform(-random_move, random_move)]
        swimmers_loc[i] = swimmers_loc[i] + swimmers_speed[i]
    swimmers_loc[:, 0] = np.clip(swimmers_loc[:, 0], 0, pool_size[0])
    swimmers_loc[:, 1] = np.clip(swimmers_loc[:, 1], 0, pool_size[1])
    return swimmers_loc, swimmers_speed

def edge(loc, pool_size):
    if abs(loc[1] - pool_size[1]) < 0.5 or abs(loc[1] - 0) < 0.5:
        return True
    else:
        return False

def under_coverage(sensors_loc, sensors_direction, loc):
    for s in range(len(sensors_loc)):
        angle = clock_wise_angle(loc - sensors_loc[s], [np.cos(sensors_direction[s]), np.sin(sensors_direction[s])])
        if abs(angle) < span/2 and math.dist(sensors_loc[s], loc) > min_range and math.dist(sensors_loc[s], loc) < max_range:
            # print(loc, sensors_loc[s])
            # print(angle, math.dist(sensors_loc[s], loc))
            return True
    return False
def even_distributed(num_sensor, pool_size):
    sensor_loc = []
    sensor_direction = []
    dis = [pool_size[0]/(num_sensor[0] + 1), pool_size[1]/(num_sensor[1] + 1)]
    for i in range(1, num_sensor[0]):
        sensor_loc.append([dis[0] * i, 0])
        sensor_direction.append(np.pi/2)
        sensor_loc.append([dis[0] * i, pool_size[1]])
        sensor_direction.append(3 * np.pi / 2)
    for i in range(1, num_sensor[1]):
        sensor_loc.append([0, dis[1] * i])
        sensor_direction.append(0)
        sensor_loc.append([pool_size[0], dis[1] * i])
        sensor_direction.append(np.pi)
    return sensor_loc, sensor_direction
# 0 no plotting, 1 plot the swimmer(dynamic), 2 plot the missed points
plt_mode = 0

pool_size = [20, 50]
num_swimmers = np.linspace(1, 50)
num_sensors = np.array([np.linspace(1, 10, 10) * 2, np.linspace(1, 10, 10) * 5])
Z = np.zeros((np.shape(num_swimmers)[0], np.shape(num_sensors)[1]))
for i in range(np.shape(num_swimmers)[0]):
    for j in range(np.shape(num_sensors)[1]):
        num_swimmer = int(num_swimmers[i])
        num_sensor = [int(num_sensors[0, j]), int(num_sensors[1, j])]
        print(num_sensor, num_swimmer)
        # init for swimmers
        swimmers_loc = np.random.random((num_swimmer, 2)) * np.array(pool_size)
        swimmers_speed = np.zeros((num_swimmer, 2))
        # init for sensors, evenly distributed
        sensor_loc, sensor_direction = even_distributed(num_sensor, pool_size)
        count = 0
        whole = num_swimmer * max_time
        if plt_mode == 1:
            plt.ion()
            points = plt.scatter(0, 1)
        for t in range(max_time):
            swimmers_loc, swimmers_speed = swimmers_move(swimmers_loc, swimmers_speed, pool_size)
            if plt_mode == 1:
                points.remove()
                points = plt.scatter(swimmers_loc[:, 0], swimmers_loc[:, 1])
                plt.pause(0.0001)
            for swimmer in swimmers_loc:
                if under_coverage(sensor_loc, sensor_direction, swimmer):
                    count = count + 1
                elif plt_mode == 2:
                    plt.scatter(swimmer[0], swimmer[1])
        Z[i, j] = (count / whole)
        if plt_mode == 2:
            plt.show()
X, Y = np.meshgrid(np.linspace(1, 10, 10)*14, num_swimmers)
print(X.shape, Y.shape, Z.shape)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, cmap='rainbow')
plt.show()
