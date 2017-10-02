import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# load the images:
filename = "./mnist_test/0.0.jpg"

img = mpimg.imread(filename)

print("data type : {} \ndata shape : {}".format(type(img), img.shape))
print(img)

# show the image:
imgplot = plt.imshow(img)
plt.show()

#-------------------------------------------------------

# read images by loop:
import numpy as np
import os

src = "./mnist_test/"
filenames = os.listdir(src)
print(filenames)

images = np.zeros((len(filenames), 28, 28), dtype=np.float32)
labels = np.zeros((len(filenames),1), dtype=np.int64)

num = 0
for each in filenames:
	img = mpimg.imread(src+each)
	images[num, :, :] = img
	if each[0] == "0":
		labels[num, :] = 0
	else:
		labels[num, :] = 1
	num += 1

print(images.shape)
print(labels)

imgplot = plt.imshow(images[-1,:,:])
plt.show()