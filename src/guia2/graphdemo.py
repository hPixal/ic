import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1 + np.exp(-z))

class Perceptron:
    def __init__(self, id):
        self.inputs = [-1]
        self.weights = np.random.rand(1);
        self.id = id
        
    def setInputs(self, inputs):
        inputs.insert(0, -1)
        self.weights = np.random.rand(len(self.inputs))
        self.inputs = inputs
        
    def addInput(self, input):
        self.inputs.append(input)
        self.weights = np.random.rand(len(self.inputs))

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
    def __init__(self):
        self.perceptrons = []
        self.perceptronConnections = {}  # Dictionary to map outputPerceptron -> [inputPerceptrons]
        self.fixedInputs = {}            # Dictionary to map perceptronID -> fixedInputID
        self.inputConnections = {}       # Dictionary to map fixedInputID -> [perceptronID]
        self.layerRegistry = {}          # Dictionary to map perceptronID -> layerNumber
        self.layersAmount = 0
        self.enableGraph = True

    def addPerceptron(self):
        id = len(self.perceptrons)
        self.perceptrons.append(Perceptron(id))
        return id

    def addFixedInput(self, id):
        self.fixedInputs[id] = Inputs(id)

    def connectFixedInput(self, fixedInputId, perceptronId):
        if fixedInputId not in self.inputConnections:
            self.inputConnections[fixedInputId] = []
            self.inputConnections[fixedInputId].append(perceptronId)  
        self.inputConnections[fixedInputId].append(perceptronId)

    def addPerceptronConnection(self, outputPerceptron, inputPerceptron):
        if outputPerceptron not in self.perceptronConnections:
            self.perceptronConnections[outputPerceptron] = []
        self.perceptronConnections[outputPerceptron].append(inputPerceptron)

    def disableGraph(self):
        self.enableGraph = False

    def enableGraph(self):
        self.enableGraph = True

    def isInputPerceptron(self, perceptronID):
        for key,value in self.perceptronConnections.items():
            if perceptronID in value:
                return True
        return False

    def isOutputPerceptron(self, perceptronID):
        if perceptronID in self.perceptronConnections.keys():
            return True
        return False
    
    def hasFixedInput(self, perceptronID):
        for key in self.inputConnections:
            if perceptronID in self.inputConnections[key]:
                return 1
        return 0

    def updateLayerStructure(self):
        for i in range(len(self.perceptrons)):
            if self.hasFixedInput(i):
                self.layerRegistry[i] = 0  # Assuming fixed inputs start at layer 0
                for j in self.perceptronConnections[i]:
                    self.recursiveLayerSearch(1, j)

    def getConnections(self, id):
        return self.perceptronConnections.get(id, [])

    def recursiveLayerSearch(self, layer, id):
        self.layerRegistry[id] = layer
        if self.isOutputPerceptron(id):
            connections = self.getConnections(id)
            for conn in connections:
                self.recursiveLayerSearch(layer + 1, conn)
                
    def setInputValue(self, fixed_input_id, value):
        if fixed_input_id in self.fixedInputs:
            self.fixedInputs[fixed_input_id].setValue(value)
                
    def getInputs(self):
        return self.fixedInputs
    
    def getPerceptrons(self):
        return self.perceptrons
    
    
    def initialize(self):                
        pass

    def recursiveForward():
        pass
    
    def plot_network(self):
        layer_dict = {}
        for perceptron_id, layer in self.layerRegistry.items():
            if layer not in layer_dict:
                layer_dict[layer] = []
            layer_dict[layer].append(perceptron_id)

        max_layer_size = max(len(layer_dict[layer]) for layer in layer_dict)
        height_step = 1 / (max_layer_size + 1)
        layer_step = 1 / (len(layer_dict) + 1)

        plt.figure(figsize=(10, 8))

        node_positions = {}

        boxCoordinates = {}
        # Plot fixed inputs as rectangles in layer 0
        for i, (input_id, perceptron_ids) in enumerate(self.inputConnections.items()):
            for _, perceptron_id in enumerate(perceptron_ids):
                y_pos = (i + 1) * height_step
                x_pos = 0
                plt.gca().add_patch(plt.Rectangle((x_pos - 0.05, y_pos - 0.0275), 0.025, 0.025, color='black'))
                boxCoordinates[input_id] = [x_pos - 0.09 + 0.0125, y_pos - 0.0275 + 0.0125]
                plt.text(x_pos - 0.05, y_pos + 0.055, f"Fixed Input {input_id}", ha='center', va='center', color='green')
                node_positions[(0, perceptron_id)] = (x_pos, y_pos)

        # Plot perceptrons as circles
        for layer, perceptrons in layer_dict.items():
            for i, perceptron_id in enumerate(perceptrons):
                y_pos = (i + 1) * height_step
                x_pos = (layer + 1) * layer_step
                plt.gca().add_patch(plt.Circle((x_pos, y_pos), 0.025, color='blue'))
                plt.text(x_pos, y_pos, f"P{perceptron_id}", ha='center', va='center', color='white')
                node_positions[(layer, perceptron_id)] = (x_pos, y_pos)

        # Draw connections between perceptrons and from fixed inputs
        for output_id, input_ids in self.perceptronConnections.items():
            if output_id in self.layerRegistry:
                output_layer = self.layerRegistry[output_id]
                for input_id in input_ids:
                    print(output_layer, output_id, input_id)
                    if input_id in self.layerRegistry:
                        input_layer = self.layerRegistry[input_id]
                        x_output, y_output = node_positions[(output_layer, output_id)]
                        x_input, y_input = node_positions[(input_layer, input_id)]
                        plt.plot([x_output, x_input], [y_output, y_input], color='black')

        for fixed_input_id, perceptron_ids in self.inputConnections.items():
            for _, perceptron_id in enumerate(perceptron_ids):
                x_fixed, y_fixed = boxCoordinates[fixed_input_id]
                perceptron_layer = self.layerRegistry[perceptron_id]
                x_perceptron, y_perceptron = node_positions[(perceptron_layer, perceptron_id)]
                plt.plot([x_fixed + 0.05, x_perceptron], [y_fixed, y_perceptron], color='red')

        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')
        plt.show()

def easy_network_test():
    nn = NeuronalNetwork()

    # Adding perceptrons and connections
    perceptron1_id = nn.addPerceptron()
    perceptron2_id = nn.addPerceptron()
    perceptron3_id = nn.addPerceptron()

    # Add fixed inputs
    nn.addFixedInput(id=0)
    nn.addFixedInput(id=1)

    # Connect fixed inputs to perceptrons
    nn.connectFixedInput(fixedInputId=1, perceptronId=perceptron1_id)
    nn.connectFixedInput(fixedInputId=0, perceptronId=perceptron1_id)
    nn.connectFixedInput(fixedInputId=1, perceptronId=perceptron2_id)
    nn.connectFixedInput(fixedInputId=0, perceptronId=perceptron2_id)

    # Add connections between perceptrons
    nn.addPerceptronConnection(outputPerceptron=perceptron1_id, inputPerceptron=perceptron3_id)
    nn.addPerceptronConnection(outputPerceptron=perceptron2_id, inputPerceptron=perceptron3_id)

    # Update the layer structure
    nn.updateLayerStructure()

    # Plot the network
    nn.plot_network()
    
def complex_network_test():
    nn = NeuronalNetwork()

    # Adding perceptrons
    perceptron_ids = [nn.addPerceptron() for _ in range(15)]  # Assume 5 inputs per perceptron

    # Add fixed inputs
    for i in range(4):
        nn.addFixedInput(i)

    # Connect fixed inputs to multiple first-layer perceptrons
    nn.connectFixedInput(fixedInputId=0, perceptronId=perceptron_ids[0])
    nn.connectFixedInput(fixedInputId=0, perceptronId=perceptron_ids[1])
    nn.connectFixedInput(fixedInputId=1, perceptronId=perceptron_ids[2])
    nn.connectFixedInput(fixedInputId=1, perceptronId=perceptron_ids[3])
    nn.connectFixedInput(fixedInputId=2, perceptronId=perceptron_ids[4])
    nn.connectFixedInput(fixedInputId=2, perceptronId=perceptron_ids[5])
    nn.connectFixedInput(fixedInputId=3, perceptronId=perceptron_ids[6])
    nn.connectFixedInput(fixedInputId=3, perceptronId=perceptron_ids[7])

    # Add complex connections between first layer and second layer perceptrons
    nn.addPerceptronConnection(outputPerceptron=perceptron_ids[0], inputPerceptron=perceptron_ids[8])
    nn.addPerceptronConnection(outputPerceptron=perceptron_ids[1], inputPerceptron=perceptron_ids[8])
    nn.addPerceptronConnection(outputPerceptron=perceptron_ids[2], inputPerceptron=perceptron_ids[9])
    nn.addPerceptronConnection(outputPerceptron=perceptron_ids[3], inputPerceptron=perceptron_ids[9])
    nn.addPerceptronConnection(outputPerceptron=perceptron_ids[4], inputPerceptron=perceptron_ids[10])
    nn.addPerceptronConnection(outputPerceptron=perceptron_ids[5], inputPerceptron=perceptron_ids[10])
    nn.addPerceptronConnection(outputPerceptron=perceptron_ids[6], inputPerceptron=perceptron_ids[11])
    nn.addPerceptronConnection(outputPerceptron=perceptron_ids[7], inputPerceptron=perceptron_ids[11])

    # Connecting second layer perceptrons to third layer perceptrons
    nn.addPerceptronConnection(outputPerceptron=perceptron_ids[8], inputPerceptron=perceptron_ids[12])
    nn.addPerceptronConnection(outputPerceptron=perceptron_ids[9], inputPerceptron=perceptron_ids[12])
    nn.addPerceptronConnection(outputPerceptron=perceptron_ids[10], inputPerceptron=perceptron_ids[13])
    nn.addPerceptronConnection(outputPerceptron=perceptron_ids[11], inputPerceptron=perceptron_ids[13])

    # Connecting third layer perceptrons to the final perceptron
    nn.addPerceptronConnection(outputPerceptron=perceptron_ids[12], inputPerceptron=perceptron_ids[14])
    nn.addPerceptronConnection(outputPerceptron=perceptron_ids[13], inputPerceptron=perceptron_ids[14])

    # Update the layer structure
    nn.updateLayerStructure()

    # Plot the network if plotting is enabled
    if nn.enableGraph:
        nn.plot_network()

# Run the easy network test
easy_network_test()

# Run the complex network test
complex_network_test()
