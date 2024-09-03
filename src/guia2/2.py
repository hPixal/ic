import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1 + np.exp(-z))


class Perceptron:
    def __init__(self, inputs, id):
        inputs.insert(0, -1)
        weights = np.random.rand(len(inputs))
        self.weights = weights
        self.inputs = inputs
        self.id = id
        

    def forward(self, inputs):
        total = np.dot(self.weights, inputs)
        return sigmoid(total)
        
        
    def backward(self, inputs, error):
        for i in range(len(inputs)):
            self.weights[i] += error * inputs[i]
        self.inputs += error
    
    def getId(self):
        return self.id
    
    def getWeights(self):
        return self.weights
    
    def getBias(self):
        return self.inputs[0]
    
class Inputs:
    
    def __init__(self, id):
        self.id = id
        self.value = 0
        
    def setValue(self, value):
        self.value = value
    
    def getValue(self):
        return self.value
        
class NeuronalNetwork:
    # Neuronal network
    
    # Works by adding perceptron and linking its inputs 
    # together, then doing forward and backward propagations
    
    def __init__(self):
        self.perceptrons = []
        self.perceptronConnections = [] # [[ outputPerceptron, inputPerceptron], ... ] TODO: Make this a map!
        
        self.fixedInputs = []    # [[ fixedInputID, perceptronID], ... ] TODO: Make this a map!
        self.inputConnections = []
        
        self.layerRegistryArray = [] # [ [perceptronID , layerNumber], ... ] TODO: Mak this a map! 
        self.layersAmount = 0
        
        self.enableGraph = 1
        
    def addPerceptron(self):
        id = len(self.perceptrons)-1
        self.perceptrons.append(Perceptron(id))
        return id
        
        
    def addFixedInputs(self, input, id):
        self.fixedInputs.append(Inputs(id))
        
    def connectFixedInputs(self, fixedInputId, perceptronId):
        self.connectFixedInputs.append([fixedInputId, perceptronId])
        
    def addPerceptronConnection(self, outputPerceptron, inputPerceptron):
        self.perceptronConnections.append([outputPerceptron, inputPerceptron])
            
    def disableGraph(self):
        self.enableGraph = 0
        
    def enableGraph(self):
        self.enableGraph = 1
                
    def isInputPerceptron(self,perceptronID):
        for i in range(len(self.perceptronConnections)):
            if(self.perceptronConnections[i][1] == perceptronID):
                return 1
        return 0

    def hasFixedInput(self,perceptronID):
        for i in range(len(self.inputConnections)):
            if(self.inputConnections[i][1] == perceptronID):
                return 1
        return 0
            
    def updateLayerStructure(self):
        for i in range(len(self.perceptrons)):
            if self.hasFixedInput(i):
                self.layerRegistryArray[i,0]
    def getConnections(self,id):
        connections = []
        for i in range(len(self.perceptronConnections)):
            if (self.perceptronConnections[i][0] == id):
                connections.append(self.perceptronConnections[i][1])
        return connections
    
    def recursiveLayerSearch(self,layer,id):
        if(not(self.isInputPerceptron(id))):
            self.layerRegistryArray.append([id,layer])
            connections = self.getConnections(id);
            for i in range(len(connections)):
                self.recursiveLayerSearch(layer+1,)
        return
    

def test():
    nn = NeuronalNetwork()
    pp = []
    
    pp.append(nn.addPerceptron())
    pp.append(nn.addPerceptron())
    pp.append(nn.addPerceptron())
    pp.append(nn.addPerceptron())
    pp.append(nn.addPerceptron())
    pp.append(nn.addPerceptron())
    
    
                