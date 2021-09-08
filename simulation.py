import numpy as np
import random
import matplotlib.pyplot as plt
import math
import time
num_swimmer = 3
num_sensor = (20, 50)
pool_size = (20, 50)
random_move = 0.1
span = 38/180 * np.pi
max_time = 1000
def swimmers_move(swimmers_loc, swimmers_speed, pool_size):
    for i in range(len(swimmers_loc)):
        if edge(swimmers_loc[i], pool_size):
            if random.random() > 0.5:
                swimmers_speed[i] = [random.uniform(-random_move, random_move), random.uniform(-random_move, random_move)]
            else:
                if swimmers_loc[i][1] > (pool_size[1]/2):
                    swimmers_speed[i] = [random.uniform(-random_move, random_move), -(random.uniform(random_move, random_move) + 1)]
                else:
                    swimmers_speed[i] = [random.uniform(-random_move, random_move), (random.uniform(-random_move, random_move) + 1)]
        else:
            swimmers_speed[i] = swimmers_speed[i] + [random.uniform(-random_move, random_move), random.uniform(-random_move, random_move)]
        swimmers_loc[i] = swimmers_loc[i] + swimmers_speed[i]
    np.clip(swimmers_loc[:, 0], 0, pool_size[0])
    np.clip(swimmers_loc[:, 1], 0, pool_size[1])
    return swimmers_loc, swimmers_speed

def edge(loc, pool_size):
    for i in range(2):
        if abs(loc[i] - pool_size[i]) < 0.5:
            return True
    return False

def under_coverage(sensors_loc, sensors_direction, loc):
    for s in range(len(sensors_loc)):
        angle = math.atan2(sensors_loc[s][1] - loc[1], sensors_loc[s][0] - loc[0])
        if angle < sensors_direction[s] + span/2 and angle > sensors_direction[s] - span/2 and math.dist(sensors_loc[s], loc) > 0.3:
            return True
    return False
def even_distributed():
    sensor_loc = []
    sensor_direction = []
    dis = [pool_size[0]/(num_sensor[0] - 1), pool_size[1]/(num_sensor[1] - 1)]
    for i in range(num_sensor[0]):
        sensor_loc.append([dis[0] * i, 0])
        sensor_direction.append(np.pi/2)
        sensor_loc.append([dis[0] * i, pool_size[1]])
        sensor_direction.append(3 * np.pi / 2)
    for i in range(num_sensor[1]):
        sensor_loc.append([0, dis[1] * i])
        sensor_direction.append(np.pi)
        sensor_loc.append([pool_size[0], dis[1] * i])
        sensor_direction.append(0)
    return sensor_loc, sensor_direction

# init for swimmers
swimmers_loc = np.random.random((num_swimmer, 2)) * np.array([20, 50])
swimmers_speed = np.zeros((num_swimmer, 2))
# init for sensors, evenly distributed
sensor_loc, sensor_direction = even_distributed()
count = 0
whole = num_swimmer * max_time
for t in range(max_time):
    swimmers_loc, swimmers_speed = swimmers_move(swimmers_loc, swimmers_speed, pool_size)
    for swimmer in swimmers_loc:
        if under_coverage(sensor_loc, sensor_direction, swimmer):
            count = count + 1
print(count / whole)
