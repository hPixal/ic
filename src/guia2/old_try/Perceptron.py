import scipy as sp
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, id: Optional[int] = None, bias: Optional[bool] = True):
        self.connections = {}   # Dictionary to map inputID -> [ inputValue, weight ]   
        self.bias = bias        # Boolean to indicate if the perceptron incorporates a bias
        self.id = id            # Unique identifier for the perceptron
        
        if bias:
            self.connections[-1] = [ -1, np.random.uniform(0, 1) ]

    
    # -------------------- MATH --------------------
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def prime_sigmoid(self,z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    # -------------------- CONFIG --------------------
    
    def setBias(self, bias):
        """
        Sets the bias of the perceptron.

        Args:
            bias (bool): A boolean indicating whether the perceptron should incorporate a bias.

        Returns:
            None
        """
        self.bias = bias
        if bias and -1 not in self.connections:
            self.connections[-1] = [ -1, np.random.rand(1) ]
        else: 
            if not bias and -1 in self.connections:
                self.connections.pop(-1)

    # -------------------- METHODS --------------------
    
    def addConnection(self, inputID, inputValue, weight):
        """
        Adds a connection to the perceptron's connections.

        Args:
            inputID (int): The ID of the input to be added.
            inputValue (float): The value of the input to be added.
            weight (float): The weight of the input to be added.

        Returns:
            None
        """
        self.connections[inputID] = [ inputValue, weight ]
    
    def removeConnection(self, inputID):
        """
        Removes a connection from the perceptron's connections.

        Args:
            inputID (int): The ID of the connection to be removed.

        Raises:
            ValueError: If the input ID is not found in the connections.

        Returns:
            None
        """
        if inputID not in self.connections:
            raise ValueError("Input ID not found in connections")
        self.connections.pop(inputID)
        
    def modifyInput(self, inputID, newValue):
        """
        Modifies the input value associated with a given input ID in the perceptron's connections.

        Args:
            inputID (int): The ID of the input to be modified.
            newValue (float): The new value to be assigned to the input.

        Raises:
            ValueError: If the input ID is not found in the connections.

        Returns:
            None
        """
        if inputID not in self.connections:
            raise ValueError("Input ID not found in connections")
        if inputID == -1: return
        self.connections[inputID][0] = newValue
        
    def modifyWeight(self, inputID, newWeight):
        """
        Modifies the weight associated with a given input ID in the perceptron's connections.

        Args:
            inputID (int): The ID of the connection to be modified.
            newWeight (float): The new weight to be assigned to the connection.

        Raises:
            ValueError: If the input ID is not found in the connections.

        Returns:
            None
        """
        if inputID not in self.connections:
            raise ValueError("Input ID not found in connections")
        self.connections[inputID][1] = newWeight
    
    def getWeights(self):
        """
        Retrieves the weights associated with the perceptron's connections.

        Returns:
            list: A list of weights corresponding to each connection.
        """
        return [ self.connections[inputID][1] for inputID in self.connections ]
    
    def getInputs(self):
        """
        Retrieves the input values associated with the perceptron's connections.

        Returns:
            list: A list of input values.
        """
        return [ self.connections[inputID][0] for inputID in self.connections ]
    
    def getOutput(self):
        """
        Retrieves the output of the perceptron by computing the dot product of the inputs and weights, 
        and then applying the sigmoid function to the result.

        Args:
            None

        Returns:
            float: The output of the perceptron.
        """
        inputs = self.getInputs()
        weights = self.getWeights()
        # print(inputs, weights);
        output = self.sigmoid(np.dot(inputs, weights))
        return output
    
    def forward(self):
        """
        This function performs a forward pass through the perceptron, 
        returning the output of the perceptron based on the current inputs and weights.
        
        Args:
            None
        
        Returns:
            float: The output of the perceptron.
        """
        return self.getOutput()
        
    