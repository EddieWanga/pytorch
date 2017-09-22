from matplotlib.finance import candlestick_ohlc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
##########################################
#              TensorBoard               #
##########################################
from logger import Logger

with open('trend_dataset_labelset.pickle', 'rb') as f:
    pickledata = pickle.load(f)

dataset = pickledata["dataset"]
labelset = pickledata["labelset"]

print(dataset.shape)
print(labelset.shape)

train_x_dataset = torch.from_numpy(dataset[:200,:,:,:])
test_x_dataset = torch.from_numpy(dataset[200:,:,:,:])
train_y_dataset = torch.from_numpy(labelset[:200])
test_y_dataset = torch.from_numpy(labelset[200:])

train_dataset = Data.TensorDataset(data_tensor=train_x_dataset, target_tensor=train_y_dataset)
test_dataset = Data.TensorDataset(data_tensor=test_x_dataset, target_tensor=test_y_dataset)

BATCH_SIZE = 32
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

class CNNModel(nn.Module):
	def __init__(self):
		super(CNNModel, self).__init__()
		# Convolution 1   # (4, 6, 6)
		self.conv1 = nn.Conv2d(in_channels=4, out_channels=20, kernel_size=3, stride=1, padding=1)
		self.relu1 = nn.ReLU()
		# MaxPooling 1    # (20, 6, 6)
		self.maxpool1 = nn.MaxPool2d(kernel_size=2) 

		# Convolution 2   # (20, 3, 3)
		self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=2, stride=1, padding=1)
		self.relu2 = nn.ReLU()
		# MaxPooling 2    # (40, 4, 4)
		self.maxpool2 = nn.MaxPool2d(kernel_size=2)
                          # (40, 2, 2)
		# Fully Connected
		self.fc = nn.Linear(40 * 2 * 2, 3)

	def forward(self, x):
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.maxpool1(x)
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.maxpool2(x)       # (batchsize, 40, 2, 2)

		# flatten:
		x = x.view(x.size(0), -1)  # (batchsize, 40* 2* 2)

		output = self.fc(x)
		return output

model = CNNModel()
# model.cuda()


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

##########################################
#              TensorBoard               #
##########################################
logger = Logger('./logs')

starttime = datetime.now()

iters = 0
for epoch in range(1000):
	for i, (images, labels) in enumerate(train_loader):
		### CPU v.s GPU
		# images = Variable(images.cuda())
		# labels = Variable(labels.cuda())
		images = Variable(images)
		labels = Variable(labels)

		# clear gradients w.r.t parameter
		optimizer.zero_grad()

		# forward pass
		output = model(images)
		_, pred = torch.max(output, 1)
		num_correct = (pred == labels).sum()
		accuracy = (pred == labels).float().mean()
		
		# calculate loss: softmax -> cross entropy
		loss = loss_func(output, labels)

		loss.backward()

		optimizer.step()

		acc = 0
		iters += 1
		if iters % 100 == 0:
			correct = 0
			total = 0
			for images, labels in test_loader:
				### CPU v.s GPU
				# images = Variable(images.cuda())
				images = Variable(images)

				output = model(images)
				_, predicted = torch.max(output.data, 1)
				total += labels.size(0)

				### CPU v.s GPU
				# correct += (predicted.cpu()==labels.cpu()).sum()
				correct += (predicted==labels).sum()

			acc = 100 * correct / total
			print("Iteration: {}. Loss: {}. Accuracy: {}".format(iters, loss.data[0], acc))			

		##########################################
		#              TensorBoard               #
		##########################################
        # ================= Log ================ #
		step = epoch * len(train_loader) + i
		
		logger.scalar_summary("lossloss", loss.data[0], step)
		logger.scalar_summary("accacc", acc, step)

endtime = datetime.now()
print("total time:", endtime-starttime)

