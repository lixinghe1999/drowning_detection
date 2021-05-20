import numpy as np
import matplotlib.pyplot as plt
former_object = [0, 20, 30, 10, 13, 80]
number_sample = 500

images = np.zeros((number_sample, former_object[2]-former_object[1]+1))
images = images[:, :, np.newaxis]
for k in range(3):
    image = np.zeros((number_sample, former_object[2]-former_object[1]+1))
    for i in range(former_object[1], former_object[2]+1):
        data = [i]*300
        image[:300, i-former_object[1]] = data
    images = np.concatenate((images, image[:, :, np.newaxis]), axis=2)
print(images.shape)