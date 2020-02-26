from sklearn import datasets
from sklearn import neural_network
from sklearn.model_selection import train_test_split
import numpy as np
from math import exp
import pandas as pd

class Layers(object):
	def __init__(self, input_neuron=1, hidden_neurons=(10,), output_neuron=1, learning_rate=0.001):
		super().__init__()
		self.num_layers 	= len(hidden_neurons) + 2
		self.neurons_num 	= [input_neuron] + [hidden_neurons[i] for i in range(len(hidden_neurons))] + [output_neuron]
		# index pertama: layer ke-berapa, index kedua: neuron ke-berapa, index ketiga: hubungan neuron tersebut dengan neuron di layer sebelumnya
		self.weights = [[] for i in range(len(hidden_neurons) + 1)]

		# index pertama: layer ke-berapa, index kedua: neuron ke-berapa
		self.biasses = [[] for i in range(len(hidden_neurons) + 1)]
		self.net 					= [[] for i in range(len(hidden_neurons) + 1)]
		self.activation 	= [[] for i in range(len(hidden_neurons) + 1)]

		self.delta_error 	= [[] for i in range(len(hidden_neurons) + 1)]
		self.delta_weight = [[] for i in range(len(hidden_neurons) + 1)]

		for i in range(len(hidden_neurons) + 1):
			# input to hidden layer
			if i == 0:

				self.biasses[i] 		= [0 for j in range(hidden_neurons[i])]
				self.net[i] 				= [0 for j in range(hidden_neurons[i])]
				self.activation[i] 	= [0 for j in range(hidden_neurons[i])]

				self.delta_error[i] = [0 for j in range(hidden_neurons[i])]
				self.delta_weight[i] = [[0 for k in range(input_neuron)] for j in range(hidden_neurons[i])]

				self.weights[i] 		= [[0 for k in range(input_neuron)] for j in range(hidden_neurons[i])]

			# hidden to output layer
			elif i == len(hidden_neurons):

				self.biasses[i] 		= [0 for j in range(output_neuron)]
				self.net[i] 				= [0 for j in range(output_neuron)]
				self.activation[i] 	= [0 for j in range(output_neuron)]

				self.delta_error[i] = [0 for j in range(output_neuron)]
				self.delta_weight[i] = [[0 for k in range(hidden_neurons[i-1])] for j in range(output_neuron)]

				self.weights[i] 		= [[0 for k in range(hidden_neurons[i-1])] for j in range(output_neuron)]

			# hidden -> hidden layer
			else:
				self.biasses[i] 		= [0 for j in range(hidden_neurons[i])]
				self.net[i] 				= [0 for j in range(hidden_neurons[i])]
				self.activation[i] 	= [0 for j in range(hidden_neurons[i])]
				self.delta_error[i] = [0 for j in range(hidden_neurons[i])]
				self.delta_weight[i] = [[0 for k in range(hidden_neurons[i-1])] for j in range(hidden_neurons[i])]
				self.weights[i] 		= [[0 for k in range(hidden_neurons[i-1])] for j in range(hidden_neurons[i])]

		self.lrate = learning_rate

	# x: features, y: target
	def feed_forward(self, x, y):
		# layer ke-n
		for layer in range(self.num_layers - 1):
			# iterasi node dalam satu LAYER
			for node in range(self.neurons_num[layer + 1]):
				# Inisialisasi value NET dengan bias
				self.net[layer][node] = self.biasses[layer][node]

				# Iterasi semua node sebelumnya untuk dapat value compute ke node sekarang
				for k in range(self.neurons_num[layer]):
					# layer == INPUT layer
					if layer == 0:
						# INPUT * bobot W(i)
						self.net[layer][node] += x[k]*self.weights[layer][node][k]
					else:
						self.net[layer][node] += self.net[layer-1][k]*self.weights[layer][node][k]
				
				# Aktivasi NET ke OUT dengan sigma
				self.activation[layer][node] = 1.0/(1 + exp(-1 * self.net[layer][node]))
		out = self.activation[-1]
		return out
	
	# Backward phase
	def backward(self, out_val, target):
		for layer in reversed(range(self.num_layers - 1)):
			for node in range(len(self.weights[layer])):
				if (layer == (self.num_layers - 2)):  # output unit
					self.delta_error[layer][node] 	= self.activation[layer][node]*(1 - self.activation[layer][node])*(target[node] - out_val[node])
				else: # hidden unit
					self.delta_error[layer][node] 	= self.activation[layer][node]*(1 - self.activation[layer][node])*sum([(self.delta_error[layer + 1][m])*self.weights[layer + 1][m][node] for m in range(len(self.weights[layer + 1]))])

	def update_delta_weight(self):
		# Update BOBOT dimulai dari layer OUTPUT, baru ke depan
		# Karena nilai yang digunakan node sekarang, merupakan ekstrak dari
		# nilai node LAYER selanjutnya

		for layer in reversed(range(self.num_layers - 1)):
			for node in range(len(self.weights[layer])):
				for k in range(len(self.weights[layer][node])):
					self.delta_weight[layer][node][k] += self.lrate * self.delta_error[layer][node]*self.activation[layer][node]

	def update_weight(self):
		for layer in reversed(range(self.num_layers - 1)):
			for node in range(len(self.weights[layer])):
				for k in range(len(self.weights[layer][node])):
					self.weights[layer][node][k] += self.delta_weight[layer][node][k]
	
	def clear_delta_weight(self):
		for layer in reversed(range(self.num_layers - 1)):
			for node in range(len(self.weights[layer])):
				for k in range(len(self.weights[layer][node])):
					self.delta_weight[layer][node][k] = 0


class myMLP(object):
	def __init__(self, hidden_layer_sizes=(10, 10,), batch_size=10, err_threshold=0.01, max_iter=2000, learning_rate=1e-05):
		super().__init__()
		self._hidden_layer_sizes = hidden_layer_sizes
		self._batch_size = batch_size
		self._err_threshold = err_threshold
		self._max_iter = max_iter
		self._lrate = learning_rate
			
	def _train(self, X, y):
		self._mlp.clear_delta_weight()
		total_err = 0.0
		
		len_X = len(X)
		i = counter = iter_counter = 0
		for index, row in X.iterrows():
			print('[{}]'.format(i), 'instance:', row, 'target:', y[i])
			# feed forward
			out = self._mlp.feed_forward(row, y[i])
			total_err += sum([(0.5*((y[i] - out_val)**2)) for out_val in out])
			print('\ttotal_err:', total_err)

			target = [1 if klas == y[i] else 0 for klas in self._unique_class]
			self._mlp.backward(out, target)
			self._mlp.update_delta_weight()
			counter += 1
			
			if (counter == self._batch_size) or (i == len_X-1):
				print('WEIGHT UPDATED')
				self._mlp.update_weight()
				self._mlp.clear_delta_weight()
				iter_counter += 1
				if (total_err < self._err_threshold) or (iter_counter == self._max_iter):
					break
				total_err = 0
				counter = 0
			i += 1
	
	def fit(self, X, y):
		self._feat_num = X.shape[1]
		self._unique_class = np.unique(y)
		self._target_class_num = len(self._unique_class)
		self._mlp = Layers(hidden_neurons=self._hidden_layer_sizes, input_neuron=self._feat_num, output_neuron=self._target_class_num, learning_rate=self._lrate)
		
		try:
			y = y[0].tolist()
		except:
			pass
		self._train(X, y)

		return self

iris = datasets.load_iris()

# bear with me for the next few steps... I'm trying to walk you through
# how my data object landscape looks... i.e. how I get from raw data 
# to matrices with the actual data I have, not the iris dataset
# put feature matrix into columnar format in dataframe
df = pd.DataFrame(data = iris.data)

# add outcome variable
df_class = pd.DataFrame(data = iris.target)

# finally, split into train-test
X_train, X_test, y_train, y_test = train_test_split(df, df_class, test_size = 0.3)

# print("iris target", y)
mMLP = myMLP(hidden_layer_sizes=(3,3,), learning_rate=1e-5)
mMLP = mMLP.fit(X_train, y_train)
