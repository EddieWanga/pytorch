import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import numpy as np

x_values = np.linspace(-1, 1, 100)
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_train = x_train**2 + random.uniform(-20, 20)

class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.hidden1 = nn.Linear(1, 10)
        self.hidden2 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        out = self.hidden2(x)
        return out

model = LR()
loss_func = nn.MSELoss()
learning_rate = 0.03
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

plt.ion()

epochs = 1000
for epoch in range(epochs):
    epoch += 1

    # convert to variables
    x = Variable(torch.from_numpy(x_train))
    y = Variable(torch.from_numpy(y_train))

    # clear gradient w.r.t. parameters 
    optimizer.zero_grad()

    # forward to get output
    prediction = model(x)

    # calculate loss
    loss = loss_func(prediction, y)

    # backward to get gradient
    loss.backward()

    # update parameters
    optimizer.step() 

    if epoch % 5 == 0:
    # plot and show learning process
        print("epoch %d, loss %.8f" % (epoch, loss.data[0]))
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=3)
        plt.text(0.5, 0, 'Step:%d Loss=%.4f' % (epoch, loss.data[0]), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()

# save the model
save_model = False
if save_model == True:
    torch.save(model.state_dict(), "nonlinear_model.pkl")

# load the model
load_model = False
if load_model is True:
    model.load_state_dict(torch.load("nonlinear_model.pkl"))
