import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
span = 40 / 180 * np.pi
distance = 10
def intersection(points1, points2):
    poly1 = Polygon(points1.reshape(3, 2)).convex_hull
    poly2 = Polygon(points2.reshape(3, 2)).convex_hull
    if not poly1.intersects(poly2):
        inter_area = 0
    else:
        inter_area = poly1.intersection(poly2).area
    return inter_area
def direction2loc(d):
    d_new = d - span/2
    x1, y1 = distance * np.array([np.cos(d_new), np.sin(d_new)])
    d_new = d + span / 2
    x2, y2 = distance * np.array([np.cos(d_new), np.sin(d_new)])
    return [x1, y1, x2, y2]

def cal_overlap(X, D):
    num_sensor = len(D)
    points = np.zeros((6, num_sensor))
    for i in range(num_sensor):
        line_2 = direction2loc(D[i])
        points[:2, i] = X[i]
        points[2:4, i] = X[i] + line_2[:2]
        points[4:, i] = X[i] + line_2[2:]
    overlap = np.zeros((num_sensor, num_sensor))
    for m in range(num_sensor):
        for n in range(num_sensor):
            overlap[m, n] = intersection(points[:, m], points[:, n])
    return overlap
def threeD(X, Y):
    Z = np.zeros(np.shape(X))
    for i in range(np.shape(X)[0]):
        for j in range(np.shape(X)[1]):
            Z[i, j] = cal_overlap(np.array([[0, 0], [0.1, 0]]), [X[i, j], Y[i, j]])[0, 1]
    return Z
D = np.linspace(0, np.pi, 50)
# D = [np.pi/4, 3*np.pi/4]
X, Y = np.meshgrid(D, D)
Z = threeD(X, Y)
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, Z, cmap = 'rainbow')
plt.show()



