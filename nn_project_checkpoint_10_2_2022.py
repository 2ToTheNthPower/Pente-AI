# -*- coding: utf-8 -*-
"""NN_Project_Checkpoint_10/2/2022.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1z0q3-anNU-BsGV2Fvvkn9SIpB1ehG9Xi
"""

!pip install compress_pickle

# !curl https://colab.chainer.org/install | sh -
import cupy as np
from matplotlib import pyplot as plt

import compress_pickle as pickle
import networkx as nx

from google.colab import drive
drive.mount('/content/drive')

def train_test_split(x, y, test_size=.25):
  data = np.append(x, y, axis=1)
  np.random.shuffle(data)

  n = data.shape[0]
  split_point = round(n*test_size)

  test_data = data[:split_point,:]
  train_data = data[split_point:, :]

  x_train = train_data[:, :-1]
  x_test = test_data[:, : -1]

  y_train = train_data[:, -1].reshape((n-split_point, 1))
  y_test = test_data[:, -1].reshape((split_point, 1))

  print(x_train.shape)
  print(y_train.shape)

  return x_train, x_test, y_train, y_test

class ANN():
  """
  x: numpy array of data
  y: numpy array of outputs
  """
  def __init__(self):
    self.X_train = None
    self.X_test = None
    self.Y_train = None
    self.Y_test = None

  def sigmoid(self, z):
    return 1/(1+np.exp(-z))

  def feedForward(self, x_inp, y_inp=None):

    # Calculate linear combination of weights and features with shape (n, hidden_neuron_count)
    self.z1 = np.matmul(x_inp, self.w1) + self.b

    # Activate z for hidden layer
    self.h1 = self.sigmoid(self.z1)

    # Generate output z
    self.z2 = np.matmul(self.h1, self.w2) + self.c

    # Activate output to get Y
    self.y_pred = self.sigmoid(self.z2)

    if y_inp is not None:
      return np.sum((self.y_pred - y_inp)**2)
    else:
      return self.y_pred

  def backprop(self, x, y):
    D_error = self.y_pred*(1 - self.y_pred)*(self.y_pred - y)
    dL_dw2 = self.h1.T @ D_error
    dL_dc = np.sum(D_error)
    dL_dw1 = (((D_error@self.w2.T) * (self.h1*(1-self.h1))).T@x).T
    dL_db = np.sum((D_error@self.w2.T) * (self.h1*(1-self.h1)), axis=0)

    self.w2 = self.w2 - self.LR*dL_dw2
    self.w1 = self.w1 - self.LR*dL_dw1
    self.b = self.b - self.LR*dL_db
    self.c = self.c - self.LR*dL_dc

  def train(self, x, y, LR=.5, num_hidden_neurons=10, epochs=100000):

    loss_list = []
    test_loss_list = []
    epoch_list = []

    self.x = x
    self.y = y

    self.test_size = .25

    if self.X_train is None:
      self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(x, y, test_size = self.test_size)

    self.n = x.shape[0]
    self.p = x.shape[1]
    self.output_dim = y.shape[1]

    self.num_hidden_neurons = num_hidden_neurons

    self.w1 = np.random.normal(size=(self.p, self.num_hidden_neurons))

    self.w2 = np.random.normal(size=(self.num_hidden_neurons, self.output_dim))
    
    self.b = np.random.normal(size=(1, self.num_hidden_neurons))

    self.c = np.random.normal(size=(1, self.output_dim))

    self.LR = LR

    for epoch in range(epochs):
      train_loss = self.feedForward(self.X_train, self.Y_train)
      average_train_loss = train_loss / (self.n - round(self.n * self.test_size))
      if epoch % 100 == 0:
        print("EPOCH:", epoch, "- TRAIN LOSS:", average_train_loss)

      self.backprop(self.X_train, self.Y_train)

      test_loss = self.feedForward(self.X_test, self.Y_test)
      average_test_loss = test_loss / round(self.n * self.test_size)

      test_loss_list.append(float(average_test_loss))
      loss_list.append(float(average_train_loss))

    epoch_list = list(range(epochs))

    return (loss_list, epoch_list, test_loss_list)

  def get_y_distributions(self):
    plt.hist([x[0] for x in self.Y_test.tolist()], bins=10)
    # plt.hist([x[0] for x in self.Y_train.tolist()])
### Test run of neural network code

# nn = ANN(x = np.array([[1,2,3,4],[-1,-2,-3,0],[3,4,-2,0],[4,4,4,4],[5,0,0,0]]), y = np.array([[1,0],[0,1],[0,0],[1,1],[0,1]]))
# nn.train(epochs=1000)

# Read in data generated by game simulation
graph = pickle.load("/content/drive/MyDrive/graph/graph.pkl")

x = []
y = []

for (node, data) in graph.nodes(data=True):
  game_state = pickle.loads(node, compression="bz2")
  x.append(np.append(game_state[0], np.array([game_state[1], game_state[2], game_state[3]])))
  if game_state[3] == 0:
    y.append(data["outcomes"] / data["visits"])
  else:
    y.append(1 - data["outcomes"] / data["visits"])

x = np.array(x)
y = np.array(y)
y = np.reshape(y, (y.shape[0], 1))

print(x[:10, :])
print(y[:10, :])

nn = ANN()
history = nn.train(x = x, y = y, LR = .001, num_hidden_neurons = 100, epochs=5000)

plt.plot(history[1], history[0])
plt.plot(history[1], history[2])
plt.xlabel("epochs")
plt.ylabel("MSE Loss")

