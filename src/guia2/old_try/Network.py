import scipy as sp
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
import Perceptron as pc

class Network:
    
    def __init__(self):
        self.perceptrons = {}           # Dictionary to map perceptronID -> Perceptron
        self.fixedInputs = {}           # Dictionary to map fixedInputID -> Input
        
        self.fixedToInputPercep = {}    # Dictionary to map fixedInputID -> [inputPerceptronIDID, ...]
        self.InputPercepToFixed = {}    # Dictionary to map inputPerceptronIDID -> [fixedInputID, ...]
        
        self.outConnections = {}        # Dictionary to map inputPerceptronID -> [outputPerceptronIDs, ...]
        self.inConnections = {}         # Dictionary to map outputPerceptronID -> [inputPerceptronIDs, ...]
        
        self.PerToPerWeights = {}       # Dictionary to map (inputPerceptronID, outputPerceptronID) -> weight
        self.FixedToPerWeights = {}     # Dictionary to map (fixedInputID, outputPerceptronID) -> weight
        
        self.layerStructure = {}        # Dictionary to map layer -> [perceptronID, ...]
        
        self.forwardOutputs = []     # Dictionary to map [ runNumber -> {inputPerceptronID -> output, ...} ]
        
        self.enableGraph = True
        self.automaticBias = True
    
    # ----------- FIXED INPUT MANAGEMENT -----------
    
    def addPerceptron(self, id: Optional[int] = None):
        """
        Adds a new perceptron to the neural network.

        Parameters:
        id (int): The unique identifier for the perceptron to be added. If not provided, a new id will be generated.

        Returns:
        int: The id of the newly added perceptron.

        Raises:
        KeyError: If a perceptron with the given id already exists in the network, or if a new id cannot be generated.
        """
        if id is None:
            id = len(self.perceptrons)
            if id not in self.perceptrons:
                self.perceptrons[id] = pc.Perceptron(id,bias=self.automaticBias)
            else:
                raise KeyError("Add id manually")
        else:
            if id not in self.perceptrons:
                self.perceptrons[id] = pc.Perceptron(id,bias=self.automaticBias)
            else:
                raise KeyError("Perceptron already exists")
        return id
    
    def addFixedInput(self, id):
        """
        Adds a new fixed input to the neural network.

        Parameters:
        id: The unique identifier for the fixed input to be added.

        Raises:
        KeyError: If a fixed input with the given id already exists in the network.
        """
        if id not in self.fixedInputs:
            self.fixedInputs[id] = 0
        else:
            raise KeyError("Fixed input already exists")

    def removeFixedInput(self, id):
        """
        Removes a fixed input from the neural network.

        Parameters:
            id (int): The unique identifier of the fixed input to be removed.

        Returns:
            None

        Raises:
            KeyError: If the fixed input with the given id does not exist in the network.
        """
        if id in self.fixedInputs:
            self.fixedInputs.pop(id)
        else:
            raise KeyError("Fixed input does not exist")

    def setInputValue(self, inputID, value):
        """
        Sets the value of a fixed input in the neural network.

        Parameters:
            inputID (int): The unique identifier of the fixed input to be updated.
            value (float): The new value to be assigned to the fixed input.

        Raises:
            KeyError: If the fixed input with the given id does not exist in the network.
        """
        if inputID not in self.fixedInputs:
            raise KeyError("Fixed input not found")
        self.fixedInputs[inputID] = value

    def setFixedToPerWeight(self, fixedInputID, percepID, value):
        """
        Sets the weight of a connection between a fixed input and a perceptron in the neural network.

        Parameters:
            fixedInputID (int): The unique identifier of the fixed input.
            percepID (int): The unique identifier of the perceptron.
            value (float): The weight value to be assigned to the connection.

        Raises:
            KeyError: If the fixed input or perceptron does not exist in the network, or if they are not connected.
        """
        if fixedInputID not in self.fixedInputs:
            raise KeyError(f"Fixed input (ID: {fixedInputID}) not found")
        if percepID not in self.perceptrons:
            raise KeyError(f"Perceptron (ID: {percepID}) not found")

        # Set weight in perceptron's connections
        if fixedInputID in self.fixedToInputPercep[percepID]:
            self.FixedToPerWeights[(fixedInputID, percepID)] = value
        else:
            raise KeyError(f"Fixed input (ID: {fixedInputID}) and Perceptron (ID: {percepID}) are not connected")
    
    def connectFixedInput(self, fixedInputId, perceptronId):
        """
        Connects a fixed input to a perceptron in the neural network.

        Parameters:
            fixedInputId (int): The unique identifier of the fixed input to be connected.
            perceptronId (int): The unique identifier of the perceptron to be connected to.

        Returns:
            None

        Raises:
            None
        """
        if fixedInputId not in self.fixedToInputPercep:
            self.fixedToInputPercep[fixedInputId] = []
        self.fixedToInputPercep[fixedInputId].append(perceptronId)

        if perceptronId not in self.InputPercepToFixed:
            self.InputPercepToFixed[perceptronId] = []
        self.InputPercepToFixed[perceptronId].append(fixedInputId)

        self.FixedToPerWeights[(fixedInputId, perceptronId)] = 0

    def disconnectFixedInput(self, fixedInputId, perceptronId):
        """
        Disconnects a fixed input from a perceptron in the neural network.

        Parameters:
            fixedInputId (int): The unique identifier of the fixed input to be disconnected.
            perceptronId (int): The unique identifier of the perceptron to be disconnected from.

        Returns:
            None

        Raises:
            None
        """
        if fixedInputId in self.fixedToInputPercep:
            self.fixedToInputPercep[fixedInputId].remove(perceptronId)
        
        if perceptronId in self.InputPercepToFixed:
            self.InputPercepToFixed[perceptronId].remove(fixedInputId)
            
    # ----------- PERCEPTRON MANAGEMENT -----------
    
    def connectPerceptrons(self, outputPerceptronID, inputPerceptronID):
        """
        Connects two perceptrons in the neural network.

        Parameters:
            outputPerceptronID (int): The unique identifier of the output perceptron.
            inputPerceptronID (int): The unique identifier of the input perceptron.

        Returns:
            None
        
        Raises:
            KeyError: If the output or input perceptrons do not exist in the network.
        """
        if outputPerceptronID not in self.perceptrons:
            raise KeyError("Output perceptron does not exist")
        if inputPerceptronID not in self.perceptrons:
            raise KeyError("Input perceptron does not exist")

        if outputPerceptronID not in self.outConnections:
            self.outConnections[outputPerceptronID] = []

        if inputPerceptronID not in self.inConnections:
            self.inConnections[inputPerceptronID] = []
            
        
        self.inConnections[inputPerceptronID].append(outputPerceptronID)
        self.outConnections[outputPerceptronID].append(inputPerceptronID)
        self.PerToPerWeights[(outputPerceptronID, inputPerceptronID)] = 0

    def disconnectPerceptrons(self, perceptronID1, perceptronID2):
        """
        Disconnects two perceptrons in the neural network.

        Parameters:
            perceptronID1 (int): The unique identifier of the first perceptron.
            perceptronID2 (int): The unique identifier of the second perceptron.

        Returns:
            None

        Raises:
            KeyError: If either of the perceptrons do not exist in the network.
        """
        if perceptronID1 not in self.perceptrons:
            raise KeyError(f"Perceptron ID:{perceptronID1} does not exist")
        if perceptronID2 not in self.perceptrons:
            raise KeyError(f"Perceptron ID:{perceptronID2} does not exist")

        if perceptronID2 in self.outConnections.get(perceptronID1, []):
            self.outConnections[perceptronID1].remove(perceptronID2)

        if perceptronID1 in self.inConnections.get(perceptronID2, []):
            self.inConnections[perceptronID2].remove(perceptronID1)

        self.PerToPerWeights.pop((perceptronID1, perceptronID2), None)

    def setPerToPerWeight(self, perceptronID1, perceptronID2, value):
        """
        Sets the weight between two perceptrons in the neural network.

        Parameters:
            perceptronID1 (int): The unique identifier of the first perceptron.
            perceptronID2 (int): The unique identifier of the second perceptron.
            value (float): The weight value to be set.

        Returns:
            None

        Raises:
            KeyError: If either of the perceptrons do not exist in the network, or if they do not have perceptron inputs.
        """
        if perceptronID1 not in self.perceptrons:
            raise KeyError(f"Perceptron ID:{perceptronID1} does not exist")
        if perceptronID2 not in self.perceptrons:
            raise KeyError(f"Perceptron ID:{perceptronID2} does not exist")
        
        # print(f"PerceptronID1: {perceptronID1}, PerceptronID2: {perceptronID2}")
        # print(f"Value: {value}")
        # print(f"map output1: {self.inConnections[perceptronID1]}")
        # print(f"map output2: {self.inConnections}")
        
        if perceptronID1 not in self.inConnections and perceptronID2 not in self.inConnections:
            raise KeyError(f"Perceptron ID:{perceptronID1} and Perceptron ID:{perceptronID2} do not have perceptron inputs")
        
        if perceptronID1 in self.inConnections:
            if perceptronID2 in self.inConnections[perceptronID1]:
                self.PerToPerWeights[(perceptronID1, perceptronID2)] = value
                self.PerToPerWeights[(perceptronID2, perceptronID1)] = value
                return
        
        if perceptronID2 in self.inConnections:
            if perceptronID1 in self.inConnections[perceptronID2]:
                self.PerToPerWeights[(perceptronID1, perceptronID2)] = value
                self.PerToPerWeights[(perceptronID2, perceptronID1)] = value
                return
            

       
            
    # ----------- UTILITY FUNCTIONS -----------
    
    def hasInputs(self, perceptronID):
        """
        Checks if a perceptron has any inputs.

        Args:
            perceptronID (str): The ID of the perceptron to check.

        Returns:
            bool: True if the perceptron has inputs, False otherwise.
        """
        return perceptronID in self.inConnections and len(self.inConnections[perceptronID]) > 0

    def hasOutputs(self, perceptronID):
        """
        Checks if a perceptron has any outputs.

        Args:
            perceptronID (str): The ID of the perceptron to check.

        Returns:
            bool: True if the perceptron has outputs, False otherwise.
        """
        return perceptronID in self.outConnections and len(self.outConnections[perceptronID]) > 0

    def getInputs(self, perceptronID):
        """
        Retrieves the input connections of a perceptron.

        Args:
            perceptronID (str): The ID of the perceptron.

        Returns:
            list: A list of input connections associated with the perceptron.
        """
        return self.inConnections.get(perceptronID, [])

    def getOutputs(self, perceptronID):
        """
        Retrieves the output connections of a perceptron.

        Args:
            perceptronID (str): The ID of the perceptron.

        Returns:
            list: A list of output connections associated with the perceptron.
        """
        return self.outConnections.get(perceptronID, [])

    def hasFixedInputs(self, perceptronID):
        """
        Checks if a perceptron has any fixed inputs.

        Args:
            perceptronID (str): The ID of the perceptron to check.

        Returns:
            bool: True if the perceptron has fixed inputs, False otherwise.
        """
        return perceptronID in self.InputPercepToFixed and len(self.InputPercepToFixed[perceptronID]) > 0

    def getFixedInputs(self, perceptronID):
        """
        Retrieves the fixed inputs associated with a perceptron.

        Args:
            perceptronID (str): The ID of the perceptron.

        Returns:
            list: A list of fixed input IDs associated with the perceptron.
        """
        return self.InputPercepToFixed.get(perceptronID, [])

    def hasInputPerceptrons(self, inputID):
        """
        Checks if an input has any associated perceptrons.

        Args:
            inputID (str): The ID of the input to check.

        Returns:
            bool: True if the input has associated perceptrons, False otherwise.
        """
        return inputID in self.fixedToInputPercep and len(self.fixedToInputPercep[inputID]) > 0

    def getInputValue(self, inputID):
        """
        Retrieves the value of a fixed input.

        Args:
            inputID (str): The ID of the fixed input.

        Returns:
            The value of the fixed input.

        Raises:
            KeyError: If the fixed input is not found.
        """
        if inputID not in self.fixedInputs:
            raise KeyError("Fixed input not found")
        return self.fixedInputs[inputID]
    
    def getNumberOfLayers(self):
        """
        Returns the number of layers in the network.

        :return: An integer representing the number of layers in the network.
        """
        return len(self.layerStructure)

    def getTotalCost(self, y_d):
        """
        Calculates the total cost of the network based on the desired outputs.

        Args:
            y_d: A dictionary containing the desired outputs for each perceptron in the output layer.

        Returns:
            The total cost of the network, calculated as the mean of the squared differences between the actual and desired outputs.

        Raises:
            ValueError: If the number of desired outputs does not match the number of perceptrons in the output layer.
        """
        output_layer = self.layerStructure[self.getNumerOfLayers() - 1]
        cost_vector = []

        if len(y_d) != len(output_layer):
            raise ValueError("The number of desired outputs must match the number of perceptrons in the last layer.")

        last_outputs = self.getLastForwardOutputs()

        for perceptron_id in output_layer:
            y = last_outputs[perceptron_id]
            y_desired = y_d.get(perceptron_id, 0)
            cost = (y - y_desired) ** 2
            cost_vector.append(cost)

        total = np.mean(cost_vector)
        return total
    
    def getForwardsOutputs(self):
        """
        Returns the last set of forward outputs from the `forwardOutputs` list.

        Returns:
            list: The last set of forward outputs.
        """
        forwardoutputs = []
        for key,value in self.forwardOutputs[-1].items():
            if key in self.layerStructure[len(self.layerStructure) - 1]:
                forwardoutputs.append(value)
        return forwardoutputs
            



    def getPerceptronInputValues(self, perceptronID):
        """
        Returns the input values for a given perceptron in the network.

        Args:
            perceptronID: The ID of the perceptron for which to retrieve input values.

        Returns:
            A list of input values for the specified perceptron.

        Raises:
            KeyError: If the perceptron with the specified ID is not found in the network.
        """
        if perceptronID not in self.perceptrons:
            raise KeyError("Perceptron not found")

        if self.hasFixedInputs(perceptronID):
            return [self.fixedInputs[inputID] for inputID in self.getFixedInputs(perceptronID)]
        else:
            return [self.getPerceptronOutputValue(inputID) for inputID in self.getInputs(perceptronID)]

    def getPerceptronOutputValue(self, perceptronID):
        """
        Returns the output value of a perceptron in the network.

        Args:
            perceptronID: The ID of the perceptron for which to retrieve the output value.

        Returns:
            The output value of the specified perceptron.

        Raises:
            KeyError: If the perceptron with the specified ID is not found in the last set of forward outputs.
        """
        if perceptronID not in self.forwardOutputs[-1]:
            raise KeyError("You haven't run forward() with the inputs of this perceptron yet")
        return self.forwardOutputs[-1][perceptronID]

    def getPerceptronWeights(self, perceptronID):
        """
        Returns the weights of a specific perceptron in the network.

        Args:
            perceptronID (int): The ID of the perceptron whose weights are to be retrieved.

        Returns:
            list: A list of weights of the specified perceptron.

        Raises:
            KeyError: If the perceptron with the specified ID is not found in the network.
        """
        if perceptronID not in self.perceptrons:
            raise KeyError("Perceptron not found")
        return self.perceptrons[perceptronID].getWeights()
    
    def getConnectionWeigths(self, inputID, outputID):
        """
        Retrieves the weights of a connection between two perceptrons in the network.

        Args:
            inputID (int): The ID of the input perceptron.
            outputID (int): The ID of the output perceptron.

        Returns:
            The weight of the connection between the specified input and output perceptrons.

        Raises:
            KeyError: If the connection between the specified input and output perceptrons is not found.
        """
        if (inputID, outputID) not in self.PerToPerWeights:
            raise KeyError("Connection not found")
        return self.PerToPerWeights[(inputID, outputID)]
    
    def getAllLastLayerOutputs(self):
        """
        Returns the last set of forward outputs from the `forwardOutputs` list.

        Returns:
            list: The last set of forward outputs.
        """
        return self.forwardOutputs[-1];


    # ----------- FORWARD AND BACKWARD -----------
    
    def forward(self):
        """
        Performs a forward pass through the network, updating the output values of all perceptrons.

        Args:
            None

        Returns:
            dict: A dictionary containing the output values of all perceptrons in the network.

        Notes:
            This function assumes that the network's layer structure has been updated and that all necessary inputs have been provided.
            If biases are not automatically added to the inputs, a warning message is printed.
        """
        if not self.automaticBias:
            print("WARNING: Biases are not automatically added to the inputs !!!!")

        self.updateLayerStructure()
        perceptron_outputs = {}
        
        for perceptron_id in self.layerStructure[0]:
            input_ids = self.getFixedInputs(perceptron_id)
            input_values = [self.fixedInputs[input_id] for input_id in input_ids]
            
            for input_id in input_ids:
                self.perceptrons[perceptron_id].addConnection(input_id,self.fixedInputs[input_id],self.FixedToPerWeights[(input_id, perceptron_id)])
            perceptron_outputs[perceptron_id] = self.perceptrons[perceptron_id].forward()

        for layer_ids in list(self.layerStructure.keys())[1:]:
            for perceptron_id in self.layerStructure[layer_ids]:
                input_ids = self.getInputs(perceptron_id)

                for input_id in input_ids:
                    self.perceptrons[perceptron_id].addConnection(input_id,perceptron_outputs[input_id],self.PerToPerWeights[(input_id, perceptron_id)])
                perceptron_outputs[perceptron_id] = self.perceptrons[perceptron_id].forward()

        self.forwardOutputs.append(perceptron_outputs)
        return perceptron_outputs
    
    def backward(self, y_d):
        """
        Performs a backward pass through the network, calculating the error gradients for all perceptrons.

        Args:
            y_d (dict): A dictionary containing the desired output values for the output layer perceptrons.

        Returns:
            dict: A dictionary containing the error gradients for all perceptrons in the network.
        """
        output_layer = self.layerStructure[self.getNumberOfLayers() - 1]
        print("LAYERS:", self.layerStructure)
        delta = {}

        for perceptron_id in output_layer:
            y = self.getPerceptronOutputValue(perceptron_id)
            y_desired = y_d.get(perceptron_id, 0)
            delta[perceptron_id] = y - y_desired

        print("DELTA:", delta)
        for i in reversed(range(self.getNumberOfLayers() - 1)):
            perceptron_ids = self.layerStructure[i]
            for perceptron_id in perceptron_ids:
                delta[perceptron_id] = 0
                print("PERCEPTRON ID:", perceptron_id)
                print("OUTPUTS", self.getOutputs(perceptron_id))
                for output_id in self.getOutputs(perceptron_id):
                    delta[perceptron_id] += delta[output_id] * self.PerToPerWeights[(output_id,perceptron_id)]
                    
        return delta
    
    # ---------------------- PLOT ----------------------

    def plot_network(self):
        """
        Plots the neural network structure.

        This function generates a visual representation of the neural network, including fixed inputs, perceptrons, and connections between them.

        Args:
            None

        Returns:
            None
        """
        max_layer_size = max(len(self.layerStructure[layer]) for layer in self.layerStructure)
        height_step = 1 / (max_layer_size + 1)
        layer_step = 1 / (len(self.layerStructure) + 1)

        plt.figure(figsize=(10, 8))

        node_positions = {}
        box_coordinates = {}

        # Plot fixed inputs as rectangles in layer 0
        for i, (input_id, perceptron_ids) in enumerate(self.fixedToInputPercep.items()):
            y_pos = (i + 1) * height_step
            x_pos = 0
            plt.gca().add_patch(plt.Rectangle((x_pos - 0.05, y_pos - 0.0275), 0.05, 0.05, color='black'))
            box_coordinates[input_id] = [x_pos - 0.05, y_pos]
            plt.text(x_pos - 0.05, y_pos + 0.055, f"Fixed Input {input_id}", ha='center', va='center', color='green')

        # Plot perceptrons as circles
        for layer, perceptrons in self.layerStructure.items():
            for i, perceptron_id in enumerate(perceptrons):
                y_pos = (i + 1) * height_step
                x_pos = (layer + 1) * layer_step
                plt.gca().add_patch(plt.Circle((x_pos, y_pos), 0.05, color='blue'))
                plt.text(x_pos, y_pos, f"P{perceptron_id}", ha='center', va='center', color='white')
                node_positions[(layer, perceptron_id)] = (x_pos, y_pos)

        # Draw connections between perceptrons
        for output_id, input_ids in self.outConnections.items():
            for input_id in input_ids:
                for layer, perceptrons in self.layerStructure.items():
                    if output_id in perceptrons:
                        x_output, y_output = node_positions[(layer, output_id)]
                        for input_layer, input_perceptrons in self.layerStructure.items():
                            if input_id in input_perceptrons:
                                x_input, y_input = node_positions[(input_layer, input_id)]
                                plt.plot([x_output, x_input], [y_output, y_input], color='black')

        # Draw connections from fixed inputs to perceptrons
        for fixed_input_id, perceptron_ids in self.fixedToInputPercep.items():
            x_fixed, y_fixed = box_coordinates[fixed_input_id]
            for perceptron_id in perceptron_ids:
                for layer, perceptrons in self.layerStructure.items():
                    if perceptron_id in perceptrons:
                        x_perceptron, y_perceptron = node_positions[(layer, perceptron_id)]
                        plt.plot([x_fixed + 0.05, x_perceptron], [y_fixed, y_perceptron], color='red')

        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')
        plt.show()

    # ---------------------- LOGICAL ----------------------


    def updateLayerStructure(self):
        """
        Updates the layer structure of the network.

        This function initializes the layer structure of the network by iterating over the perceptrons and
        categorizing them based on whether they have fixed inputs. It starts with layer level 0 and adds
        perceptrons with fixed inputs to it. Then, it recursively updates the layer structure for subsequent
        layers.

        Parameters:
            None

        Returns:
            None
        """
        layer_level = 0
        self.layerStructure = {}
        self.layerStructure[layer_level] = []

        for perceptron in self.perceptrons:
            if self.hasFixedInputs(perceptron):
                self.layerStructure[layer_level].append(perceptron)

        for layerZeroPerceptronID in self.layerStructure[0]:
            if self.hasOutputs(layerZeroPerceptronID):
                self.recursiveUpdateLayer(1, self.outConnections[layerZeroPerceptronID])
        return

    def recursiveUpdateLayer(self, layer, sameLayerPerceptrons):
        """
        Recursively updates the layer structure of the network.

        This function takes a layer and a list of perceptrons in that layer as input, and updates the layer structure
        by adding the perceptrons to the corresponding layer. If a perceptron has outputs, it recursively calls itself
        to update the next layer.

        Parameters:
            layer (int): The current layer being updated.
            sameLayerPerceptrons (list): A list of perceptron IDs in the current layer.

        Returns:
            None
        """
        if layer not in self.layerStructure:
            self.layerStructure[layer] = []

        for sameLayerPerceptronID in sameLayerPerceptrons:
            if sameLayerPerceptronID not in self.layerStructure[layer]:
                self.layerStructure[layer].append(sameLayerPerceptronID)
                if self.hasOutputs(sameLayerPerceptronID):
                    self.recursiveUpdateLayer(layer + 1, self.outConnections[sameLayerPerceptronID])
        return

    def updateWeights(self): 
        """
        Updates the weights of the perceptrons in the network.

        Iterates over the PerToPerWeights dictionary and sets the weight of each perceptron
        to the corresponding value in the dictionary.

        Parameters:
            None

        Returns:
            None
        """
        for key1, key2 in self.PerToPerWeights:
            self.perceptrons[key1].setWeight(key2, self.PerToPerWeights[key1][key2])
            
    
    # ---------------------- DEBUGGING ----------------------

    def enableAutomaticBiases(self):
        """
        Enables automatic bias for all perceptrons in the network.

        This function sets the `automaticBias` attribute of the network to `True` and then iterates over each perceptron
        in the `perceptrons` list. For each perceptron, it calls the `setBias` method with the `automaticBias` value to
        enable automatic bias.

        Parameters:
            None

        Returns:
            None
        """
        self.automaticBias = True
        for perceptron in self.perceptrons:
            perceptron.setBias(self.automaticBias)
        
    def disableAutomaticBiases(self):
        """
        Disables the automatic bias for all perceptrons in the network.

        This function sets the `automaticBias` attribute of the network to `False` and then iterates over each perceptron
        in the `perceptrons` list. For each perceptron, it calls the `setBias` method with the `automaticBias` value to
        disable automatic bias.

        Parameters:
            None

        Returns:
            None
        """
        self.automaticBias = False
        for perceptron in self.perceptrons:
            perceptron.setBias(self.automaticBias)

    def printFixedInputs(self):
        """
        Prints and returns a formatted string containing all fixed inputs in the network.
        
        Parameters:
            None
        
        Returns:
            str: A formatted string containing all fixed inputs in the network.
        """
        printString = "Fixed Inputs: {"
        for input_id, value in self.fixedInputs.items():
            printString += f"\n  Fixed Input ID: {input_id}, Value: {value}"
        printString += "\n}\n"
        print(printString)
        return printString

    def printPerceptronValues(self):
        if not self.forwardOutputs:
            return "Can't provide perceptron values, no forward output yet.\n"
        printString = "Perceptron Values: {"
        last_layer_outputs = self.forwardOutputs[-1]
        for perceptron_id in self.layerStructure[len(self.layerStructure) - 1]:
            value = last_layer_outputs.get(perceptron_id, 'N/A')
            printString += f"\n  Perceptron ID: {perceptron_id}, Value: {value}"
        printString += "\n}\n"
        print(printString)
        return printString
    
    def printLastResults(self):
        if not self.forwardOutputs:
            return "Can't provide last results, no forward output yet.\n"
        last = self.forwardOutputs[-1]
        printString = "Last Results: {"
        for perceptron_id, value in last.items():
            printString += f"\n  Perceptron ID: {perceptron_id}, Value: {value}"
        printString += "\n}\n"
        print(printString)
        return printString

    def printPerceptronFixedInputs(self, perceptronID):
        input_ids = self.getFixedInputs(perceptronID)
        printString = f"\nFixed Inputs of Perceptron: {perceptronID} are: "
        printString += "{ "
        for input_id in input_ids:
            value = self.fixedInputs.get(input_id, 'N/A')
            printString += f"\n  Input ID: {input_id}, Value: {value}"
        printString += "\n}\n"
        print(printString)
        return printString

    def printPerceptronInputs(self, perceptronID):
        input_ids = self.getInputs(perceptronID)
        printString = f"\nInputs of Perceptron: {perceptronID} are: "
        printString += "{ "
        for input_id in input_ids:
            value = self.getInputValue(input_id)
            printString += f"\n  Input ID: {input_id}, Value: {value}"
        printString += "\n}\n"
        print(printString)
        return printString

    def printCurrentPerceptronWeights(self, perceptronID):
        perceptron = self.perceptrons.get(perceptronID)
        if perceptron is None:
            return f"Perceptron with ID {perceptronID} does not exist.\n"
        weights = perceptron.getWeights()
        printString = f"\nCurrent Weights of Perceptron (ID: {perceptronID}) are: "
        printString += "{ "
        for weight in weights:
            printString += f"\n  Weight: {weight},"
        printString += "\n}\n"
        print(printString)
        return printString

    def printAllWeights(self):
        printString = "All Weights: {"
        for perceptron_id, perceptron in self.perceptrons.items():
            weights = perceptron.getWeights()
            printString += f"\n  Perceptron ID: {perceptron_id}, Weights: {weights}"
        printString += "\n}\n"
        print(printString)
        return printString

    def printPerceptronInConnections(self, perceptronID):
        input_ids = self.getInputs(perceptronID)
        printString = f"Inputs of Perceptron: {perceptronID} are: "
        printString += "{ "
        for input_id in input_ids:
            printString += f"\n  Input ID: {input_id}"
        printString += "\n}\n"
        print(printString)
        return printString

    def printPerceptronOutConnections(self, perceptronID):
        output_ids = self.getOutputs(perceptronID)
        printString = f"Outputs of Perceptron: {perceptronID} are: "
        printString += "{ "
        for output_id in output_ids:
            value = self.getPerceptronOutputValue(output_id)
            printString += f"\n  Perceptron (ID: {perceptronID}) Outputs to Perceptron (ID: {output_id}), Value: {value}"
        printString += "\n}\n"
        print(printString)
        return printString

    def printPerceptronConnections(self):
        printString = "Connections: {"
        for out_id, in_ids in self.outConnections.items():
            for in_id in in_ids:
                value = self.getPerceptronOutputValue(in_id) if self.forwardOutputs else 'N/A'
                printString += f"\n  Perceptron (ID: {out_id}) Outputs to Perceptron (ID: {in_id}), Value: {value}"
        printString += "\n}\n"
        print(printString)
        return printString

    def dumpCurrentNetworkInfoToFile(self, filename="netinfo.txt"):
        with open(filename, "w") as file:
            file.write(self.printFixedInputs())
            file.write(self.printPerceptronValues())
            file.write(self.printLastResults())
            file.write(self.printAllWeights())
            file.write(self.printPerceptronConnections())
            
