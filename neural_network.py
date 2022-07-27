import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from nnfs.datasets import vertical_data
import matplotlib.pyplot as plt

nnfs.init()

class Data_Gen():
	def __init__(self, size):
		self.size = size

	def spiral_data(self):
		X, y = spiral_data(samples=100, classes=self.size)
		return X, y

	def vertical_data(self):
		X, y = vertical_data(samples=100, classes=self.size)
		return X, y

class Layer_Dense:
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))

	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
	def forward(self, inputs):
		self.output = np.maximum(0, inputs)

# review page 102-104 again
class Activation_Softmax:
	def forward(self, inputs):
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		probablities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
		self.output = probablities

class Loss:
	def calculate(self, output, y):
		sample_losses = self.forward(output, y)
		data_loss = np.mean(sample_losses)
		return data_loss

# review this
class Loss_CategoricalCrossEntropy(Loss):
	def forward(self, y_pred, y_true):
		samples = len(y_pred)

		y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

		if len(y_true.shape) == 1:
			correct_confidences = y_pred_clipped[
				range(samples),
				y_true
			]

		elif len(y_true.shape) == 2:
			correct_confidences = np.sum(
				y_pred_clipped*y_true,
				axis=1
			)

		negative_log_likelihoods = -np.log(correct_confidences)
		return negative_log_likelihoods

def main():

	# initialize
	X,y = Data_Gen(3).vertical_data()

	dense1 = Layer_Dense(2, 3)
	dense2 = Layer_Dense(3, 3)
	activation1 = Activation_ReLU()
	activation2 = Activation_Softmax()
	loss_function = Loss_CategoricalCrossEntropy()

	# helper variables
	lowest_loss = 99999
	best_dense1_weights = dense1.weights.copy()
	best_dense2_weights = dense2.weights.copy()
	best_dense1_biases = dense1.biases.copy()
	best_dense2_biases = dense2.biases.copy()

	output = []

	it_range1 = []
	it_range2 = []

	for it in range(10000):
		dense1.weights += 0.05*np.random.randn(2, 3)
		dense1.biases += 0.05*np.random.randn(1, 3)
		dense2.weights += 0.05*np.random.randn(3, 3)
		dense2.biases += 0.05*np.random.randn(1, 3)

		# network actions
		dense1.forward(X)
		activation1.forward(dense1.output)
		dense2.forward(activation1.output)
		activation2.forward(dense2.output)
		#print(activation2.output[:5])

		# loss and accuracy
		loss = loss_function.calculate(activation2.output, y)

		predictions = np.argmax(activation2.output, axis=1)
		output = predictions
		accuracy = np.mean(predictions == y)

		if loss < lowest_loss:
			#print(f'#{it}, l={loss}, a={accuracy}')
			best_dense1_weights = dense1.weights.copy()
			best_dense2_weights = dense2.weights.copy()
			best_dense1_biases = dense1.biases.copy()
			best_dense2_biases = dense2.biases.copy()
			lowest_loss = loss

			it_range1.append(it)
			it_range2.append(loss)

			fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
			fig.set_size_inches(18.5, 10.5)
			fig.suptitle(f'{it}: loss: {loss}, accuracy: {accuracy}')
			ax1.set_title('Target')
			ax2.set_title('Prediction')
			ax3.set_title('Loss by Iteration')
			ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
			ax2.scatter(X[:, 0], X[:, 1], c=predictions, cmap='brg')
			ax3.plot(it_range1, it_range2)
			plt.show(block=False)
			plt.pause(0.001)
			plt.close()

		else:
			dense1.weights = best_dense1_weights.copy()
			dense2.weights = best_dense2_weights.copy()
			dense1.biases = best_dense1_biases.copy()
			dense2.biases = best_dense2_biases.copy()

main()

# NOTES
# reread chapter 5 to understand loss

