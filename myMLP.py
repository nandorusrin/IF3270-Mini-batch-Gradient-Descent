from sklearn import datasets
from sklearn import neural_network
from sklearn.model_selection import train_test_split
import numpy as np
from math import exp
import pandas as pd
import matplotlib.pyplot as plt
import random


class Layers(object):
	def __init__(self, input_neuron=1, hidden_neurons=(10,), output_neuron=1, learning_rate=0.001, use_momentum=False):
		super().__init__()
		random.seed()	# default seed: current system time
		self.num_layers 	= len(hidden_neurons) + 2
		self.neurons_num 	= [input_neuron] + [hidden_neurons[i] for i in range(len(hidden_neurons))] + [output_neuron]
		# index pertama: layer ke-berapa, index kedua: neuron ke-berapa, index ketiga: hubungan neuron tersebut dengan neuron di layer sebelumnya
		self.weights = [[] for i in range(len(hidden_neurons) + 1)]

		# index pertama: layer ke-berapa, index kedua: neuron ke-berapa
		self.biasses = [[] for i in range(len(hidden_neurons) + 1)]
		self.net 					= [[] for i in range(len(hidden_neurons) + 1)]
		self.activation 	= [[] for i in range(len(hidden_neurons) + 1)]

		self.delta_biasses 	= [[] for i in range(len(hidden_neurons) + 1)]
		self.delta_error 	= [[] for i in range(len(hidden_neurons) + 1)]
		self.delta_weight = [[] for i in range(len(hidden_neurons) + 1)]

		for i in range(len(hidden_neurons) + 1):
			# input to hidden layer
			if i == 0:

				self.biasses[i] 		= [random.uniform(0.0, 0.1) for j in range(hidden_neurons[i])]
				self.net[i] 				= [0 for j in range(hidden_neurons[i])]
				self.activation[i] 	= [0 for j in range(hidden_neurons[i])]

				self.delta_biasses[i] = [0 for j in range(hidden_neurons[i])]
				self.delta_error[i] = [0 for j in range(hidden_neurons[i])]
				self.delta_weight[i] = [[0 for k in range(input_neuron)] for j in range(hidden_neurons[i])]

				self.weights[i] 		= [[random.uniform(0.0, 0.1) for k in range(input_neuron)] for j in range(hidden_neurons[i])]

			# hidden to output layer
			elif i == len(hidden_neurons):

				self.biasses[i] 		= [random.uniform(0.0, 0.1) for j in range(output_neuron)]
				self.net[i] 				= [0 for j in range(output_neuron)]
				self.activation[i] 	= [0 for j in range(output_neuron)]

				self.delta_biasses[i] = [0 for j in range(output_neuron)]
				self.delta_error[i] = [0 for j in range(output_neuron)]
				self.delta_weight[i] = [[0 for k in range(hidden_neurons[i-1])] for j in range(output_neuron)]

				self.weights[i] 		= [[random.uniform(0.0, 0.1) for k in range(hidden_neurons[i-1])] for j in range(output_neuron)]

			# hidden -> hidden layer
			else:
				self.biasses[i] 		= [random.uniform(0.0, 0.1) for j in range(hidden_neurons[i])]
				self.net[i] 				= [0 for j in range(hidden_neurons[i])]
				self.activation[i] 	= [0 for j in range(hidden_neurons[i])]
				
				self.delta_biasses[i] = [0 for j in range(hidden_neurons[i])]
				self.delta_error[i] = [0 for j in range(hidden_neurons[i])]
				self.delta_weight[i] = [[0 for k in range(hidden_neurons[i-1])] for j in range(hidden_neurons[i])]
				self.weights[i] 		= [[random.uniform(0.0, 0.1) for k in range(hidden_neurons[i-1])] for j in range(hidden_neurons[i])]

		self.lrate = learning_rate
		self.use_momentum = use_momentum

	def set_momentum(self, momentum):
		self.momentum = momentum

	def print_layers(self):
		print('biasses')
		for i in range(len(self.biasses)):
			print(self.biasses[i])
		print('\nweights')
		for i in range(len(self.weights)):
	 		print(self.weights[i])
		print()

	# x: features, y: target
	def feed_forward(self, x):
		# layer ke-n
		for layer in range(self.num_layers - 1):
			# iterasi node dalam satu LAYER
			for node in range(self.neurons_num[layer + 1]):
				# Inisialisasi value NET dengan bias
				self.net[layer][node] = float(self.biasses[layer][node])

				# Iterasi semua node sebelumnya untuk dapat value compute ke node sekarang
				for k in range(self.neurons_num[layer]):
					# layer == INPUT layer
					if layer == 0:
						# INPUT * bobot W(i)
						self.net[layer][node] += x[k]*self.weights[layer][node][k]
					else:
						self.net[layer][node] += self.activation[layer-1][k]*self.weights[layer][node][k]
				
				# Aktivasi NET ke OUT dengan sigma
				self.activation[layer][node] = 1.0/(1 + exp(-1 * self.net[layer][node]))
		out = self.activation[-1]
		return out
	
	# Backward phase
	def backward(self, out_val, target_probs):
		for layer in reversed(range(self.num_layers - 1)):
			for node in range(len(self.weights[layer])):
				if (layer == (self.num_layers - 2)):  # output unit
					self.delta_error[layer][node] 	= self.activation[layer][node]*(1 - self.activation[layer][node])*(target_probs[node] - out_val[node])
				else: # hidden unit
					self.delta_error[layer][node] 	= self.activation[layer][node]*(1 - self.activation[layer][node])*sum([(self.delta_error[layer + 1][m])*self.weights[layer + 1][m][node] for m in range(len(self.weights[layer + 1]))])

	def update_delta_weight(self):
		# Update BOBOT dimulai dari layer OUTPUT, baru ke depan
		# Karena nilai yang digunakan node sekarang, merupakan ekstrak dari
		# nilai node LAYER selanjutnya

		for layer in reversed(range(self.num_layers - 1)):
			for node in range(len(self.weights[layer])):
				self.delta_biasses[layer][node] += self.lrate*self.delta_error[layer][node]*1
				for k in range(len(self.weights[layer][node])):
					self.delta_weight[layer][node][k] += self.lrate * self.delta_error[layer][node]*self.activation[layer][node]
					

	def update_weight(self):
		for layer in reversed(range(self.num_layers - 1)):
			for node in range(len(self.weights[layer])):
				if self.use_momentum:
					self.biasses[layer][node] += self.momentum * self.biasses[layer][node]
				self.biasses[layer][node] += self.delta_biasses[layer][node]
				for k in range(len(self.weights[layer][node])):
					if self.use_momentum:
						self.weights[layer][node][k] += self.momentum * self.weights[layer][node][k]
					self.weights[layer][node][k] += self.delta_weight[layer][node][k]
		if self.use_momentum:
			self.momentum /= 2
					
	
	def clear_delta_weight(self):
		for layer in reversed(range(self.num_layers - 1)):
			for node in range(len(self.weights[layer])):
				self.delta_biasses[layer][node] = 0
				for k in range(len(self.weights[layer][node])):
					self.delta_weight[layer][node][k] = 0
	


class myMLP(object):
	def __init__(self, hidden_layer_sizes=(10, 10,), batch_size=100, err_threshold=0.01, max_iter=200, learning_rate=0.001, use_momentum=False, momentum=0.9, early_stopping=True):
		super().__init__()
		self._hidden_layer_sizes = hidden_layer_sizes
		self._batch_size = batch_size
		self._err_threshold = err_threshold
		self._max_iter = max_iter
		self._lrate = learning_rate
		self.use_momentum = use_momentum
		self.momentum = momentum
			
	def _train(self, X, y):
		self._mlp.clear_delta_weight()
		total_err = 0.0
		error_hist = []
		done = False
		len_X = len(X)
		for iter_counter in range(self._max_iter):
			i = counter = 0
			for index, row in X.iterrows():
				# print('[{}]'.format(i), 'instance:', row, 'target:', y[i])
				# feed forward
				out = self._mlp.feed_forward(row)
				
				target_probs = [1 if klas == y[i] else 0 for klas in self._unique_class]
				total_err += 0.5*sum([((out[out_idx] - target_probs[out_idx])**2) for out_idx in range(len(out))])

				self._mlp.backward(out, target_probs)
				self._mlp.update_delta_weight()
				counter += 1
				
				if (counter == self._batch_size) or (i == len_X-1):
					error_hist.append(total_err)
					# print('WEIGHT UPDATED', total_err)
					self._mlp.update_weight()
					self._mlp.clear_delta_weight()
					if (total_err < self._err_threshold and iter_counter > 0):
						done = True
						break
					if (counter == self._batch_size and not (i == len_X-1)):
						total_err = 0
					counter = 0
				i += 1
			if done:
				break

		print('Last total err:', total_err)
		total_err_df = pd.DataFrame(data=error_hist)
		ax = total_err_df.plot(kind='line')

		plt.show()

		self._mlp.print_layers()

	def fit(self, X, y):
		self._feat_num = X.shape[1]
		self._unique_class = np.unique(y)
		self._target_class_num = len(self._unique_class)
		self._mlp = Layers(hidden_neurons=self._hidden_layer_sizes, input_neuron=self._feat_num, output_neuron=self._target_class_num, learning_rate=self._lrate, use_momentum=self.use_momentum)
		if self.use_momentum:
			self._mlp.set_momentum(self.momentum)
		
		try:
			y = y[0].tolist()
		except:
			pass
		self._train(X, y)

		return self
	
	def predict(self, x_test):
		predict_value = []
		for i, x in X_test.iterrows():
			value = self._mlp.feed_forward(x)
			# print(value)
			maxIdx = np.argmax(value)
			predict_value.append(self._unique_class[maxIdx])
			#(index, predictedValue, probability)
		return(predict_value)

	def accuracy(self, x_test, y_test):
		try:
			y_test = y_test[0].tolist()
		except:
			pass

		predicts = self.predict(x_test)
		print(predicts)
		len_data = len(predicts)
		count = 0
		for p_idx in range(len_data):
			if (predicts[p_idx] == y_test[p_idx]):
				count += 1
		return count/float(len_data)


# IRIS dataset
iris = datasets.load_iris()

df = pd.DataFrame(data=iris.data)
df_norm = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df_class = pd.DataFrame(data=iris.target)

# finally, split into train-test
X_train, X_test, y_train, y_test = train_test_split(df_norm, df_class, test_size=0.2)
mMLP = myMLP(hidden_layer_sizes=(3,3), learning_rate=0.001, max_iter=10, batch_size=10, err_threshold=0.1, use_momentum=True, early_stopping=False)
mMLP = mMLP.fit(X_train, y_train)
print('Accuracy:', mMLP.accuracy(X_test, y_test) * 100, '%')

# Referensi hidden layer
# The architecture and the units of the input, hidden and output layers in sklearn are decribed as below:

# The number of input units will be the number of features (in general +1 node for bias)
# For multiclass classification the number of output units will be the number of labels
# Try a single hidden layer first
# The more the units in a hidden layer the better, try the same as the number of input features.
# Some general rules about the hidden layer are the following based on this paper: Approximating Number of Hidden layer neurons in Multiple Hidden Layer BPNN Architecture by Saurabh Karsoliya.

# In general:

# The number of hidden layer neurons are 2/3 (or 70% to 90%) of the size of the input layer.
# The number of hidden layer neurons should be less than twice of the number of neurons in input layer.
# The size of the hidden layer neurons is between the input layer size and the output layer size.
