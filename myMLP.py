from sklearn import datasets
from sklearn import neural_network
import numpy as np
from math import exp

class Layers(object):
  def __init__(self, input_neuron=1, hidden_neurons=(10,), output_neuron=1, learning_rate=0.001):
    super().__init__()
    self.weights = [[] for i in range(len(hidden_neurons) + 1)]
    self.biasses = [[] for i in range(len(hidden_neurons) + 1)]
    self.temp = [[] for i in range(len(hidden_neurons) + 1)]
    self.num_layers = len(hidden_neurons) + 2

    self.neurons_num = [input_neuron] + [hidden_neurons[i] for i in range(len(hidden_neurons))] + [output_neuron]

    for i in range(len(hidden_neurons) + 1):
      if i == 0:  # input to hidden layer
        self.biasses[i] = [0 for j in range(hidden_neurons[i])]
        self.temp[i] = [0 for j in range(hidden_neurons[i])]
        self.delta[i] = [0 for j in range(hidden_neurons[i])]
        self.weights[i] = [
          [0 for k in range(input_neuron)] for j in range(hidden_neurons[i])
        ]
      elif i == len(hidden_neurons):  # hidden to output layer
        self.biasses[i] = [0 for j in range(output_neuron)]
        self.temp[i] = [0 for j in range(output_neuron)]
        self.delta[i] = [0 for j in range(output_neuron)]
        self.weights[i] = [
          [0 for k in range(hidden_neurons[i-1])] for j in range(output_neuron)
        ]
      else:
        self.biasses[i] = [0 for j in range(hidden_neurons[i])]
        self.temp[i] = [0 for j in range(hidden_neurons[i])]
        self.delta[i] = [0 for j in range(hidden_neurons[i])]
        self.weights[i] = [
          [0 for k in range(hidden_neurons[i-1])] for j in range(hidden_neurons[i])
        ]

    print('nn:', self.neurons_num)
    print(self.weights)
    print(self.biasses)
    self.lrate = learning_rate


  def feed_forward(self, x, y):
    print('x, y', x, y)
    for i in range(self.num_layers-1):
      for j in range(len(self.temp[i])):
        # init with net value
        self.temp[i][j] = self.biasses[i][j]
        for k in range(self.neurons_num[i]):
          if i == 0:  # from input
            self.temp[i][j] += x[k]*self.weights[i][j][k]
          else:
            self.temp[i][j] += self.temp[i-1][k]*self.weights[i][j][k]
        # compute net to out
        self.temp[i][j] = 1.0/(1 + exp(-1 * self.temp[i][j]))
    out = self.temp[-1]
    return out
  
  # TODO: backprop mechanism
  def backprop(self, out_val, target):
    for i in reversed(range(self.num_layers-1)):
      for j in range(len(self.weights[i])):
        for k in range(len(self.weights[i][j])):
          if i == self.num_layers-1:  # output unit
            self.delta[i][j] = self.temp[i][j]*(1-self.temp[i][j])*(target[j]-out_val[j])
            # self.weights[j][k] -= self.lrate * (-1*(target[j] - out_val[j])*out_val[j]*(1-out_val[j])) * out_val[j]
          else: # hidden unit
            self.delta[i][j] = self.temp[i][j]*(1-self.temp[i][j])*sum([(self.delta[i+1][m])*self.weights[i+1][m] for m in range(len(self.weights[i+1]))])


class myMLP(object):
  def __init__(self, hidden_layer_sizes=(10,), batch_size=200):
    super().__init__()
    self._hidden_layer_sizes = hidden_layer_sizes
    self._batch_size = batch_size
      
  def _train(self, X, y):
    for i in range(len(X)):
      # feed forward
      out = self._mlp.feed_forward(X[i], y[i])
      total_err = sum([(0.5*((y[i] - out_val)**2)) for out_val in out])
      print(total_err)
      
      # backward phase
      self._mlp.backprop(out)
  
  def fit(self, X, y):
    self._feat_num = X.shape[1]
    self._target_class_num = len(np.unique(y))
    self._mlp = Layers(hidden_neurons=self._hidden_layer_sizes, input_neuron=self._feat_num, output_neuron=self._target_class_num)

    self._train(X, y)

    return self

iris = datasets.load_iris()
X = iris.data
y = iris.target
mMLP = myMLP(hidden_layer_sizes=(2,))
mMLP = mMLP.fit(X, y)
