from sklearn import datasets
from sklearn import neural_network
import numpy as np
from math import exp

class Layers(object):
  def __init__(self, input_neuron=1, hidden_neurons=(10,), output_neuron=1, learning_rate=0.001):
    super().__init__()
    # index pertama: layer ke-berapa, index kedua: neuron ke-berapa, index ketiga: hubungan neuron tersebut dengan neuron di layer sebelumnya
    self.weights = [[] for i in range(len(hidden_neurons) + 1)]
    # index pertama: layer ke-berapa, index kedua: neuron ke-berapa
    self.biasses = [[] for i in range(len(hidden_neurons) + 1)]
    
    self.net = [[] for i in range(len(hidden_neurons) + 1)]
    self.activation = [[] for i in range(len(hidden_neurons) + 1)]
    self.delta = [[] for i in range(len(hidden_neurons) + 1)]
    self.num_layers = len(hidden_neurons) + 2

    self.neurons_num = [input_neuron] + [hidden_neurons[i] for i in range(len(hidden_neurons))] + [output_neuron]

    for i in range(len(hidden_neurons) + 1):
      if i == 0:  # input to hidden layer
        self.biasses[i] = [0 for j in range(hidden_neurons[i])]
        self.net[i] = [0 for j in range(hidden_neurons[i])]
        self.activation[i] = [0 for j in range(hidden_neurons[i])]
        self.delta[i] = [0 for j in range(hidden_neurons[i])]
        self.weights[i] = [
          [0 for k in range(input_neuron)] for j in range(hidden_neurons[i])
        ]
      elif i == len(hidden_neurons):  # hidden to output layer
        self.biasses[i] = [0 for j in range(output_neuron)]
        self.net[i] = [0 for j in range(output_neuron)]
        self.activation[i] = [0 for j in range(output_neuron)]
        self.delta[i] = [0 for j in range(output_neuron)]
        self.weights[i] = [
          [0 for k in range(hidden_neurons[i-1])] for j in range(output_neuron)
        ]
      else:
        self.biasses[i] = [0 for j in range(hidden_neurons[i])]
        self.net[i] = [0 for j in range(hidden_neurons[i])]
        self.activation[i] = [0 for j in range(hidden_neurons[i])]
        self.delta[i] = [0 for j in range(hidden_neurons[i])]
        self.weights[i] = [
          [0 for k in range(hidden_neurons[i-1])] for j in range(hidden_neurons[i])
        ]

    # print('nn:', self.neurons_num)
    # print(self.weights)
    # print(self.biasses)
    self.lrate = learning_rate

  # x: features, y: target
  def feed_forward(self, x, y):
    print('x, y', x, y)
    for layer in range(self.num_layers-1):
      for node in range(self.neurons_num[layer+1]):
        # init with net value
        self.net[layer][node] = self.biasses[layer][node]
        for k in range(self.neurons_num[layer]):
          if layer == 0:  # from input
            self.net[layer][node] += x[k]*self.weights[layer][node][k]
          else:
            self.net[layer][node] += self.net[layer-1][k]*self.weights[layer][node][k]
        # compute net to out
        self.activation[layer][node] = 1.0/(1 + exp(-1 * self.net[layer][node]))
    out = self.activation[-1]
    return out
  
  # TODO: backward mechanism
  def backward(self, out_val, target):
    for layer in reversed(range(self.num_layers-1)):
      for node in range(len(self.weights[layer])):
        if layer == self.num_layers-2:  # output unit
          # print(target, out_val);
          self.delta[layer][node] = self.activation[layer][node]*(1-self.activation[layer][node])*(target[node]-out_val[node])
        else: # hidden unit
          # print("error", layer, node);
          self.delta[layer][node] = self.activation[layer][node]*(1-self.activation[layer][node])*sum(
            [(self.delta[layer+1][m])*self.weights[layer+1][m][node] for m in range(len(self.weights[layer+1]))]
          )
    print("delta", self.delta)

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
      # print(total_err)
      target = [1 if klas == y[i] else 0 for klas in self._unique_class]
      # backward phase
      self._mlp.backward(out, target)
  
  def fit(self, X, y):
    self._feat_num = X.shape[1]
    self._unique_class = np.unique(y)
    self._target_class_num = len(self._unique_class)
    self._mlp = Layers(hidden_neurons=self._hidden_layer_sizes, input_neuron=self._feat_num, output_neuron=self._target_class_num)
    
    self._train(X, y)

    return self

iris = datasets.load_iris()
X = iris.data
y = iris.target
# print("iris target", y)
mMLP = myMLP(hidden_layer_sizes=(2,))
mMLP = mMLP.fit(X, y)
