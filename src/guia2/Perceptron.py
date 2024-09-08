import scipy as sp
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, id: Optional[int] = None):
        self.inputs = {}   
        self.weights = {}
        if id is not None:
            self.id = id
        else:
            self.id = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def prime_sigmoid(self,z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def setInputs(self, inputs, inputPerceptronIDs,bias=True):
        if len(inputs) != len(self.weights) and len(self.weights) != 0:
            raise Exception(f"The number of inputs in Perceptron (ID: {self.id}) must match the number of weights \n number of inputs: {str(len(inputs))} \n number of weights: {str(len(self.weights))} \n Inputs given: {str(inputs)}")
        
        for i in range(len(inputs)):
            self.inputs[inputPerceptronIDs[i]] = inputs[i]
        
        # Add bias input if needed (assuming it's always -1 for simplicity)
        if bias:
            self.inputs[-1] = 0 # Bias input at position 0
            
        if len(self.weights) == 0:
            for i in range(len(inputs)):
                self.weights[inputPerceptronIDs[i]] = np.random.rand(1);
                
    def setWeights(self, weights, inputPerceptronIDs,bias=True):
        if len(weights) != len(self.inputs) and len(self.inputs) != 0:
            raise Exception(f"The number of weights in Perceptron (ID: {self.id}) must match the number of inputs \n number of inputs: {str(len(self.inputs))} \n number of weights: {str(len(weights))} \n Weights given: {str(weights)}")
        
        for i in range(len(weights)):
            self.weights[inputPerceptronIDs[i]] = weights[i]
        
        # Add bias input if needed (assuming it's always -1 for simplicity)
        if bias:
            self.weights[-1] = np.rand()  # Bias input at position 0
            
        if len(self.weights) == 0:
            for i in range(len(weights)):
                self.weights[inputPerceptronIDs[i]] = np.random.rand(1);
            
    def getWeights(self):
        return self.weights
    
    def getBias(self):
        return self.weights[-1]
    
    def getID(self):
        return self.id
    
    def forward(self):
        if len(self.weights) != len(self.inputs):
            raise Exception(f"The number of weights must match the number of inputs in Perceptron: {self.id} \n number of inputs: {str(len(self.inputs))} \n number of weights: {str(len(self.weights))} \n Weights given: {str(self.weights)} \n Inputs given: {str(self.inputs)}")
        
        if len(self.weights) == 0:
            print(f"WARNING: Perceptron (ID: {self.id}) has inputs and weights length of 0!!")
            return 0
        
        total = np.dot(self.weights, self.inputs)
        return self.sigmoid(total)
    
