import numpy as np

# Class is just like in java minus having to declare the fields
class MLP:

    # Constructor
    ## '=' after the input is the default value if not provided
    ### The num_hidden contains a list of the number of neurons in each layer
    def __init__(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2):

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        # Gets a list where each item in the list represents then number of neurons in a layer
        ## i.e. [3, 3, 5, 2] for the default values
        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]
        
        
        # initiate random weights
        self.weights = []
        # For each layer...
        for i in range(len(layers) - 1):
            # Make a weight matrix (2d array) 
            # The # rows wanted is the current layer and the # cols wanted is the number of neurons in the layer after
            w =  np.random.rand(layers[i], layers[i + 1])

            self.weights.append(w)

    # Forward propogates inputs all the way to the outputs
    def forward_propagate(self, inputs):
        
        # For the first layer the activations are basically the inputs
        activations = inputs

        # Loop through all the weights
        for w in self.weights:

            # Calculate net inputs for given layer
            ## This takes the activations (inputs) and multiplies them by their respective weight
            net_inputs = np.dot(activations, w)

            # Calculate the activations
            activations = self._sigmoid(net_inputs)

        return activations
    
    def _sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))


# Create an MLP
mlp = MLP()

# Create some inputs
inputs = np.random.rand(mlp.num_inputs)

# Perform the forward prop
outputs = mlp.forward_propagate(inputs)

# Print results
print("The network input is: {}".format(inputs))
print("The network output is: {}".format(outputs))
