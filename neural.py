import random, math

class Connection:
	def __init__(self):
		self.weight = Connection.randomWeight()
		self.deltaWeight = 0.0
		print('Made connection with starting weight of {0}' .format(self.weight))
	
	@staticmethod
	def randomWeight():
		return random.random()


class Neuron:
	eta = 0.15 # 0.0 - slow learner, 0.2 - medium learner, 1.0 - reckless learner
	alpha = 0.5 # 0.0 - no momentum, 0.5 - moderate momentum
	
	def __init__(self, numOutputs, index):
		self.outputWeights = []
		for iWeights in range(numOutputs):
			self.outputWeights.append(Connection())
		self.outputVal = 0.0
		self.gradient = 0.0
		self.index = index
		print('Made neuron {0}' .format(self.index))
	
	def feedForward(self, prevLayer):
		sumPrev = 0.0
		# sum the previous layer's outputs (which are our inputs) weighted by the connection
		# include the bias node from the previous layer
		for prevNeuron in prevLayer.neurons:
			sumPrev += prevNeuron.outputVal * prevNeuron.outputWeights[self.index].weight
		
		self.outputVal = Neuron.transferFunction(sumPrev)
	
	def calcOutputGradients(self, targetVal):
		delta = targetVal - self.outputVal
		self.gradient = delta * Neuron.transferFunctionDerivative(self.outputVal)
	
	def calcHiddenGradients(self, nextLayer):
		dow = self.sumDOW(nextLayer)
		self.gradient = dow * Neuron.transferFunctionDerivative(self.outputVal)
	
	def updateInputWeights(self, prevLayer):
		# the weights to be updated are the the connection container in the neurons in the preceding layer
		for iNeuron in range(len(prevLayer.neurons)):
			neuron = prevLayer.neurons[iNeuron]
			oldDeltaWeight = neuron.outputWeights[self.index].deltaWeight
			# individual input, magnified by the gradient and train rate
			# also add momentum = a fraction of the previous delta weight
			newDeltaWeight = Neuron.eta * neuron.outputVal * self.gradient + Neuron.alpha * oldDeltaWeight
			neuron.outputWeights[self.index].deltaWeight = newDeltaWeight
			neuron.outputWeights[self.index].weight += newDeltaWeight
	
	def sumDOW(self, nextLayer):
		sumErr = 0.0
		# sum our contributions of the errors at the nodes we feed
		for iNeuron in range(len(nextLayer.neurons)-1):
			sumErr += self.outputWeights[iNeuron].weight * nextLayer.neurons[iNeuron].gradient
		
		return sumErr
			
	@staticmethod
	def transferFunction(x):
		# there are many possible transfer functions for a neuron
		# could be simple as a threshold or just linear
		# this one is more realistic though
		return math.tanh(x)
	
	@staticmethod
	def transferFunctionDerivative(x):
		# the derivate of the transfer function
		return 1-math.tanh(x)**2


class Layer:
	def __init__(self, numNeurons, numOutputsPerNeuron):
		self.neurons = []
		for iNeuron in range(numNeurons):
			self.neurons.append(Neuron(numOutputsPerNeuron, iNeuron))
		
		# force the bias nodes's output to 1.0. it's the last neuron created above
		self.neurons[-1].outputVal = 1.0


class Net:
	def __init__(self, topology):
		self.layers = []
		self.recentAverageError = 0.0
		numLayers = len(topology)
		for iLayer in range(numLayers):
			print('Building layer {0}' .format(iLayer))
			# number of neurons based on topology + 1 bias neuron on each layer
			numNeuronsPerLayer = topology[iLayer]+1
			# the number of "real" neurons on the next layer (the outputs of each neuron)
			numNeuronsNextLayer = 0 if iLayer == numLayers-1 else topology[iLayer+1]
			self.layers.append(Layer(numNeuronsPerLayer, numNeuronsNextLayer))
	
	def feedForward(self, inputVals):
		# assing the input values into the input neurons
		for iNeuron in range(len(inputVals)):
			self.layers[0].neurons[iNeuron].outputVal = inputVals[iNeuron]
		
		# forward propagation starting from first hidden layer up to output layer
		for iLayer in range(1, len(self.layers)):
			for neuron in self.layers[iLayer].neurons[:-1]:
				# pass the previous layer to each neurons feedForward process
				neuron.feedForward(self.layers[iLayer-1])
		
	def backProp(self, targetVals):
		outputLayer = self.layers[-1]
		self.error = 0.0
		
		# (optional) calculate the overall net error (RMS of output neurons error) excluding the bias neuron
		# this is just for us to know how good the network performes
		for iNeuron in range(len(outputLayer.neurons)-1):
			delta = targetVals[iNeuron] - outputLayer.neurons[iNeuron].outputVal
			self.error += delta**2
		self.error /= len(outputLayer.neurons)-1 # averaged error squared
		self.error = math.sqrt(self.error) # RMS
		# recent average measurements
		self.smoothError = 0.5
		self.recentAverageError = (self.recentAverageError * self.smoothError + self.error * (1-self.smoothError))		
		
		# calculate output layer gradients
		for iNeuron in range(len(outputLayer.neurons)-1):
			outputLayer.neurons[iNeuron].calcOutputGradients(targetVals[iNeuron])
		
		# calculate gradients on hidden layers
		for iLayer in range(len(self.layers)-2, 0, -1):
			hiddenLayer = self.layers[iLayer]
			nextLayer = self.layers[iLayer+1]
			for neuron in hiddenLayer.neurons:
				neuron.calcHiddenGradients(nextLayer)
		
		# for all layers from outputs to first hidden layer, update connection weights
		for iLayer in range(len(self.layers)-1, 0, -1):
			layer = self.layers[iLayer]
			prevLayer = self.layers[iLayer-1]
			for neuron in layer.neurons[:-1]:
				neuron.updateInputWeights(prevLayer)
	
	def getResults(self):
		resultVals = []
		for neuron in self.layers[-1].neurons[:-1]:
			resultVals.append(neuron.outputVal)
		
		return resultVals
	
	def reportRecentAverageError(self):
		print('Net recent average error: {0}' .format(self.recentAverageError))
		

# open training data file
#
# the first line contains the number of inputs and outputs. For example:
# Inputs: 2 Outputs: 1
# each line contains the values for inputs and outputs, whitespace separated. For example (XOR):
# 0 0 0
# 0 1 1
# 1 0 1
# 1 1 0
with open('training_data_norm.txt') as f:
	trainOptions = f.readline().split()
	numInputs = (int)(trainOptions[trainOptions.index('Inputs:')+1])
	numOutputs = (int)(trainOptions[trainOptions.index('Outputs:')+1])
	
	# build topology
	topology = [numInputs, 2, numOutputs] # number of neurons on each layer from input -> output
	# in this case, we have numInputs input neurons on the first layer, 2 neurons on 1 hidden layer, and numOutputs neurons on the output layer
	# hidden layers form the abstraction in a neuron network
	
	# make network from this topology
	myNet = Net(topology)
	
	# training the network
	iLine = 0
	for line in f:
		iLine += 1
		# read data
		trainData = line.split()
		inputVals = list(map(float, trainData[:numInputs]))
		targetVals = list(map(float, trainData[-numOutputs:]))
		
		myNet.feedForward(inputVals)
		myNet.backProp(targetVals)
		
		print('\nPass: {0}' .format(iLine))
		print('Inputs: {0}\nTargets: {1}' .format(inputVals, targetVals))
		
		resultVals = myNet.getResults()
		
		print('Network results: {0}' .format(resultVals))
		myNet.reportRecentAverageError()
