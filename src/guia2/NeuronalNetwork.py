import scipy as sp
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
import Perceptron as pc

class NeuronalNetwork:
    
    def __init__(self):
        self.perceptrons = {}           # Dictionary to map perceptronID -> Perceptron
        self.fixedInputs = {}           # Dictionary to map fixedInputID -> Input
        
        self.fixedToInputPercep = {}    # Dictionary to map fixedInputID -> [inputPerceptronIDID, ...]
        self.InputPercepToFixed = {}    # Dictionary to map inputPerceptronIDID -> [fixedInputID, ...]
        
        self.outConnections = {}        # Dictionary to map inputPerceptronID -> [outputPerceptronIDs, ...]
        self.inConnections = {}         # Dictionary to map outputPerceptronID -> [inputPerceptronIDs, ...]
        
        self.layerStructure = {}        # Dictionary to map layer -> [perceptronID, ...]
        
        self.forwardOutputs = []     # Dictionary to map [ runNumber -> {inputPerceptronID -> output, ...} ]
        
        self.enableGraph = True
        self.automaticBias = True
    
    # ----------- ADDERS AND REMOVERS -----------
    
    def addPerceptron(self, id: Optional[int] = None):
        """
        Adds a new perceptron to the neural network.
        
        Parameters:
        id (int): The unique identifier for the perceptron to be added.
        
        Returns:
        None
        
        Raises:
            KeyError: If a perceptron with the given id already exists in the network.
        """
        if id is None:
            id = len(self.perceptrons)
            if id not in self.perceptrons:
                self.perceptrons[id] = pc.Perceptron(id)
            else:
                raise KeyError("Add id manually")
        else:
            if id not in self.perceptrons:
                self.perceptrons[id] = pc.Perceptron(id)
            else:
                raise KeyError("Perceptron already exists")
        return id
    
    def removePerceptron(self, id):
        """
        Removes a perceptron from the neural network.
        
        Parameters:
        id (int): The unique identifier of the perceptron to be removed.
        
        Returns:
        None
        
        Raises:
            KeyError: If the perceptron with the given id does not exist in the network.
        """
        if id in self.perceptrons:
            self.perceptrons.pop(id)
        else:
            raise KeyError("Perceptron does not exist")
        
    def addFixedInput(self, id):
        """
        Adds a new fixed input to the neural network.
        
        Parameters:
        id (int): The unique identifier for the fixed input to be added.
        
        Returns:
        None
        
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
        Sets the value of a fixed input.

        Args:
            inputID (int): The unique identifier of the fixed input.
            value (float): The new value of the fixed input.
        """
        if inputID not in self.fixedInputs:
            raise KeyError("Fixed input not found")
        
        self.fixedInputs[inputID] = value
            
    def setPerceptronWeights(self, perceptronID, values):
        """
        Sets the weights of a perceptron in the neural network.

        Parameters:
            perceptronID (int): The unique identifier of the perceptron to set the weights for.
            values (list): A list of values representing the new weights of the perceptron.

        Raises:
            KeyError: If the perceptron with the given ID does not exist in the network.
        """
        if perceptronID not in self.perceptrons:
            raise KeyError(f"Perceptron (ID: {perceptronID}) not found")
        self.perceptrons[perceptronID].setWeights(values,self.getInputs(perceptronID))
        
    # ----------- CONNECTIONS AND DISCONNECTIONS -----------
    
    def connectFixedInput(self, fixedInputId, perceptronId):
        """
        Connects a fixed input to a perceptron in the neural network.

        Parameters:
        fixedInputId (int): The unique identifier of the fixed input to be connected.
        perceptronId (int): The unique identifier of the perceptron to be connected to.

        Returns:
        None

        Raises:
            KeyError: If the connection cannot be established.
        """
        if fixedInputId not in self.fixedToInputPercep:
            self.fixedToInputPercep[fixedInputId] = []
            self.fixedToInputPercep[fixedInputId].append(perceptronId)
        else:
            self.fixedToInputPercep[fixedInputId].append(perceptronId)
        
        if perceptronId not in self.InputPercepToFixed:
            self.InputPercepToFixed[perceptronId] = []
            self.InputPercepToFixed[perceptronId].append(fixedInputId)
        else:
            self.InputPercepToFixed[perceptronId].append(fixedInputId)
        
    def disconnectFixedInput(self, fixedInputId, perceptronId):
        """
        Disconnects a fixed input from a perceptron in the neural network.

        Parameters:
            fixedInputId (int): The unique identifier of the fixed input to be disconnected.
            perceptronId (int): The unique identifier of the perceptron to be disconnected from.

        Returns:
            None
            
        Raises:
            KeyError: If the fixed input or perceptron does not exist in the network.
        """
        self.fixedToInputPercep[fixedInputId].remove(perceptronId)
        self.InputPercepToFixed[perceptronId].remove(fixedInputId)
        
        
    def addPerceptronConnection(self, outputPerceptronID, inputPerceptronID):
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
            self.outConnections[outputPerceptronID].append(inputPerceptronID)
        else:
            self.outConnections[outputPerceptronID].append(inputPerceptronID)
        
        if inputPerceptronID not in self.inConnections:
            self.inConnections[inputPerceptronID] = []
            self.inConnections[inputPerceptronID].append(outputPerceptronID)
        else:
            self.inConnections[inputPerceptronID].append(outputPerceptronID)
            
    def disconnectPerceptronConnection(self, perceptronID1, perceptronID2 ):
        """
        Disconnects two perceptrons in the neural network.

        Parameters:
            perceptronID1 (int): The unique identifier of the first perceptron.
            perceptronID2 (int): The unique identifier of the second perceptron.

        Returns:
            None

        Raises:
            KeyError: If either perceptron does not exist in the network or if the perceptrons are not connected.
        """
        if perceptronID1 not in self.perceptrons:
            raise KeyError("Output perceptron does not exist")
        if perceptronID2 not in self.perceptrons:
            raise KeyError("Input perceptron does not exist")
        
        if perceptronID2 not in self.outConnections[perceptronID1]:
            raise KeyError("Perceptrons not connected")
        if perceptronID1 not in self.outConnections[perceptronID2]:
            raise KeyError("Perceptrons not connected")
        
        if perceptronID1 not in self.inConnections[perceptronID2]:
            self.inConnections[perceptronID2].remove(perceptronID1)
            self.outConnections[perceptronID1].remove(perceptronID2)
        
        if perceptronID2 not in self.inConnections[perceptronID1]:
            self.inConnections[perceptronID1].remove(perceptronID2)
            self.outConnections[perceptronID2].remove(perceptronID1)
        
    # ----------- CONSULTATIONS -----------
        
    # Perceptron connections
        
    def hasInputs(self, perceptronID):
        """
        Check if a perceptron has any inputs.
        
        Args:
            perceptronID (int): The unique identifier of the perceptron.
            
        Returns:
            bool: True if the perceptron has inputs, False otherwise.
        """
        if perceptronID in self.inConnections:
            return len(self.inConnections[perceptronID]) > 0
        else:
            return False
        
    def hasOutputs(self, perceptronID):
        """
        Checks if a perceptron has any outputs.
        
        Args:
            perceptronID (int): The unique identifier of the perceptron.
            
        Returns:
            bool: True if the perceptron has outputs, False otherwise.
        """
        if perceptronID in self.outConnections:
            return len(self.outConnections[perceptronID]) > 0
        else:
            return False
        
    def getInputs(self, perceptronID):
        """
        Retrieves the input connections of a perceptron.

        Args:
            perceptronID (int): The unique identifier of the perceptron.

        Returns:
            list: A list of input connections for the specified perceptron. If the perceptron has no inputs, an empty list is returned.
        """
        if self.hasInputs(perceptronID):
            return self.inConnections[perceptronID]
        else:
            return []
    
    def getOutputs(self, perceptronID):
        """
        Retrieves the output connections of a perceptron.

        Args:
            perceptronID (int): The unique identifier of the perceptron.

        Returns:
            list: A list of output connections for the specified perceptron. If the perceptron has no outputs, an empty list is returned.
        """
        if self.hasOutputs(perceptronID):
            return self.outConnections[perceptronID]
        else:
            return []
    
    # Fixed conections
    
    def hasFixedInputs(self, perceptronID):
        """
        Checks if a perceptron has any fixed inputs.

        Args:
            perceptronID (int): The unique identifier of the perceptron.

        Returns:
            bool: True if the perceptron has fixed inputs, False otherwise.
        """
        if perceptronID in self.InputPercepToFixed:
            return len(self.InputPercepToFixed[perceptronID]) > 0
        else:
            return False
    
    def getFixedInputs(self, perceptronID):
        """
        Retrieves the fixed input connections of a perceptron.

        Args:
            perceptronID (int): The unique identifier of the perceptron.

        Returns:
            list: A list of fixed input connections for the specified perceptron. If the perceptron has no fixed inputs, an empty list is returned.
        """
        if self.hasFixedInputs(perceptronID):
            return self.InputPercepToFixed[perceptronID]
        else:
            return []
    
    def hasInputPerceptrons(self, inputID):
        """
        Checks if a fixed input has any connected input perceptrons.

        Args:
            inputID (int): The unique identifier of the fixed input.

        Returns:
            bool: True if the fixed input has connected input perceptrons, False otherwise.
        """
        if inputID in self.fixedToInputPercep:
            return len(self.fixedToInputPercep[inputID]) > 0
        else:
            return False
        
    # Getting values
    def getInputValue(self, inputID):
        """
        Retrieves the value of a fixed input.

        Args:
            inputID (int): The unique identifier of the fixed input.

        Returns:
            float: The value of the fixed input.
        """
        if inputID not in self.fixedInputs:
            raise KeyError("Fixed input not found")
        
        return self.fixedInputs[inputID]

    def getPerceptronInputValues(self, perceptronID):
        """
        Retrieves the input values of a specific perceptron in the network.

        Args:
            perceptronID (int): The unique identifier of the perceptron whose input values are to be retrieved.

        Returns:
            list: A list of input values for the specified perceptron.

        Raises:
            KeyError: If the perceptron is not found or if the perceptron has no inputs.
        """
        inputValues = []
        if perceptronID not in self.perceptrons:
            raise KeyError("Perceptron not found")
        if self.hasFixedInputs(perceptronID):
            for inputID in self.getFixedInputs(perceptronID):
                inputValues.append(self.fixedInputs[inputID])
        else:
            if perceptronID not in self.inConnections:
                raise KeyError("Perceptron has no inputs")
            for inputID in self.inConnections[perceptronID]:
                inputValues.append(self.getPerceptronOutputValue(inputID))
        return inputValues;
    
    def getPerceptronOutputValue(self, perceptronID):
        """
        Retrieves the output value of a specific perceptron after a forward pass.

        Args:
            perceptronID (int): The unique identifier of the perceptron whose output value is to be retrieved.

        Returns:
            float: The output value of the specified perceptron.

        Raises:
            KeyError: If the perceptron has not been run through a forward pass yet.
        """
        if perceptronID not in self.forwardOutputs[len(self.forwardOutputs) - 1]:
            raise KeyError("You havent run forward() with the inputs of this perceptron yet")
        
        return self.forwardOutputs[len(self.forwardOutputs)-1][perceptronID]
    
    
    def getWeights(self, perceptronID):
        """
        Retrieves the weights of a specific perceptron in the network.

        Args:
            perceptronID (int): The unique identifier of the perceptron whose weights are to be retrieved.

        Returns:
            list: A list of weights for the specified perceptron.

        Raises:
            KeyError: If the perceptron is not found.
        """
        if perceptronID not in self.perceptrons:
            raise KeyError("Perceptron not found")
        
        return self.perceptrons[perceptronID].getWeights()
        
    # ------------------ LOGICAL ------------------
    
    def updateLayerStructure(self):
        """
        Updates the layer structure of the neural network.
        
        This function initializes the layer registry and populates it with perceptrons 
        based on their connections. It starts by identifying the input layer and then 
        recursively updates the layer structure for subsequent layers.
        
        Parameters:
            None
        
        Returns:
            None
        """
        layer_level = 0
        
        self.layerRegistry = {}
        self.layerRegistry[layer_level] = []
        
        for perceptron in self.perceptrons:
            if self.hasFixedInputs(perceptron):
                self.layerRegistry[layer_level].append(perceptron)
        
        for layerZeroPerceptronID in self.layerRegistry[0]:
            if self.hasOutputs(layerZeroPerceptronID):
                self.recursiveUpdateLayer(1, self.outConnections[layerZeroPerceptronID])
        return
    
    def recursiveUpdateLayer(self, layer, sameLayerPerceptrons):
        """
        Recursively updates the layer structure of the neural network.

        This function is called to update the layer registry based on the connections
        between perceptrons. It starts by iterating over the perceptrons in the
        `sameLayerPerceptrons` list. If a perceptron is not already in the current
        layer, it is added and checked for outputs. If the perceptron has outputs,
        the function is called recursively to update the layer structure for the
        next layer.

        Parameters:
            layer (int): The current layer level.
            sameLayerPerceptrons (list): A list of perceptron IDs in the current layer.

        Returns:
            None
        """
         # Ensure the layer exists in layerRegistry
        if layer not in self.layerRegistry:
            self.layerRegistry[layer] = []

        for sameLayerPerceptronID in sameLayerPerceptrons:
            if sameLayerPerceptronID not in self.layerRegistry[layer]:
                self.layerRegistry[layer].append(sameLayerPerceptronID)
                if self.hasOutputs(sameLayerPerceptronID):
                    self.recursiveUpdateLayer(layer + 1, self.outConnections[sameLayerPerceptronID])
        return
                    
    def forward(self):
        """
        This function propagates the input through the neural network, 
        processing each layer of perceptrons sequentially. It starts by 
        processing the input layer, then moves on to subsequent layers, 
        using the outputs of previous layers as inputs to the next. The 
        function returns a dictionary of perceptron outputs for the last 
        layer of the network.

        Parameters:
            None

        Returns:
            dict: A dictionary where the keys are the perceptron IDs and 
            the values are their corresponding outputs.
        """
        if not self.automaticBias:
            print("WARNING: Biases are not automatically added to the inputs !!!!")
        
        perceptron_outputs = {}
        
        perceptron_ids = self.layerRegistry[0]
            
        for perceptron_id in perceptron_ids:
            
            input_ids = self.getFixedInputs(perceptron_id)
            input_values = []
            
            for input_id in input_ids:
                input_values.append(self.fixedInputs[input_id])
                
            self.perceptrons[perceptron_id].setInputs(input_values,input_ids, bias=self.automaticBias)
            perceptron_outputs[perceptron_id] = self.perceptrons[perceptron_id].forward();
            
            
            perceptron_outputs[perceptron_id] = self.perceptrons[perceptron_id].forward();
        
        for layer_ids in list(self.layerRegistry.keys())[1:]:
        # Process the perceptron_ids in layer 1 and beyond
            perceptron_ids = self.layerRegistry[layer_ids]
            for perceptron_id in perceptron_ids:
                
                input_ids = self.getInputs(perceptron_id)
                input_values = []
                
                for input_id in input_ids:
                    input_values.append(perceptron_outputs[input_id])
                    
                self.perceptrons[perceptron_id].setInputs(input_values, input_ids, bias=self.automaticBias)
                perceptron_outputs[perceptron_id] = self.perceptrons[perceptron_id].forward();
            
        self.forwardOutputs.append(perceptron_outputs)
        return perceptron_outputs
    
    def getTotalCost(self,y_d):
        output_layer = self.layerRegistry[self.getNumerOfLayers()-1] 
        costVector = []
        
        if len(y_d) != len(output_layer):
            raise ValueError("The number of desired outputs must match the number of perceptrons in the last layer.")
        
        for perceptron_id, output in self.getLastForwardOutputs().items():
            y = output
            y_desired = y_d[perceptron_id]
            cost = pow(2,y-y_d)
            costVector.append(cost)
        
        total = np.average(costVector)
        
        return total;
        
    def getLastForwardOutputs(self):
        return self.forwardOutputs[len(self.forwardOutputs)-1]

    def backward(self, y_d):
        omegas = {}
        last_forward_outputs = self.forwardOutputs[len(self.forwardOutputs)-1]
        for i in reversed(self.getNumerOfLayers()-1):
            if i == self.getNumerOfLayers()-1:
                perceptron_ids = self.layerRegistry[i]
                for perceptron_id in perceptron_ids:
                    y = self.getLastForwardOutputs()[perceptron_id]
                    y_desired = y_d[perceptron_id]
                    error = y_desired - y
                    omegas[perceptron_id] = error * 1 / len(self.getPerceptronInputValues(perceptron_id)) * self.prime_sigmoid(last_forward_outputs[perceptron_id])
            else:
                perceptron_ids = self.layerRegistry[i]
                for perceptron_id in perceptron_ids:
                    omegas[perceptron_id] = 0
                    for output_id in self.outConnections[perceptron_id]:
                        omegas[perceptron_id] += omegas[output_id] * self.perceptrons[output_id].weights[perceptron_id] # <-- fix
                    
                 
            
    def prime_sigmoid(self,z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    

    def getNumerOfLayers(self):
        return len(self.layerRegistry)
    # ------------------ PLOTTING ------------------
    
    def plot_network(self):
        """
        Plots the neural network structure with perceptrons as circles and fixed inputs as rectangles.
        The connections between perceptrons and from fixed inputs are also drawn.

        Parameters:
            None

        Returns:
            None
        """

        max_layer_size = max(len(self.layerRegistry[layer]) for layer in self.layerRegistry)
        height_step = 1 / (max_layer_size + 1)
        layer_step = 1 / (len(self.layerRegistry) + 1)

        plt.figure(figsize=(10, 8))

        node_positions = {}
        boxCoordinates = {}

        # Plot fixed inputs as rectangles in layer 0
        for i, (input_id, perceptron_ids) in enumerate(self.fixedToInputPercep.items()):
            y_pos = (i + 1) * height_step
            x_pos = 0
            plt.gca().add_patch(plt.Rectangle((x_pos - 0.05, y_pos - 0.0275), 0.05, 0.05, color='black'))
            boxCoordinates[input_id] = [x_pos - 0.05, y_pos]
            plt.text(x_pos - 0.05, y_pos + 0.055, f"Fixed Input {input_id}", ha='center', va='center', color='green')

        # Plot perceptrons as circles
        for layer, perceptrons in self.layerRegistry.items():
            for i, perceptron_id in enumerate(perceptrons):
                y_pos = (i + 1) * height_step
                x_pos = (layer + 1) * layer_step
                plt.gca().add_patch(plt.Circle((x_pos, y_pos), 0.05, color='blue'))
                plt.text(x_pos, y_pos, f"P{perceptron_id}", ha='center', va='center', color='white')
                node_positions[(layer, perceptron_id)] = (x_pos, y_pos)

        # Draw connections between perceptrons
        for output_id, input_ids in self.outConnections.items():
            for input_id in input_ids:
                for layer, perceptrons in self.layerRegistry.items():
                    if output_id in perceptrons:
                        x_output, y_output = node_positions[(layer, output_id)]
                        for input_layer, input_perceptrons in self.layerRegistry.items():
                            if input_id in input_perceptrons:
                                x_input, y_input = node_positions[(input_layer, input_id)]
                                plt.plot([x_output, x_input], [y_output, y_input], color='black')

        # Draw connections from fixed inputs to perceptrons
        for fixed_input_id, perceptron_ids in self.fixedToInputPercep.items():
            for perceptron_id in perceptron_ids:
                x_fixed, y_fixed = boxCoordinates[fixed_input_id]
                for layer, perceptrons in self.layerRegistry.items():
                    if perceptron_id in perceptrons:
                        x_perceptron, y_perceptron = node_positions[(layer, perceptron_id)]
                        plt.plot([x_fixed + 0.05, x_perceptron], [y_fixed, y_perceptron], color='red')

        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')
        plt.show()


    # ------------------ DEBUGGING ------------------
    
    def enableAutomaticBiases(self):
        self.automaticBias = True
    
    def disableAutomaticBiases(self):
        self.automaticBias = False
    
    def printFixedInputs(self):
        """
        Prints and returns a string representation of the fixed inputs in the neural network.

        This function iterates over each fixed input in the network, retrieves its ID and value, 
        and appends a string representation of the fixed input's ID and value to the output string.

        Returns:
            str: A string representation of the fixed inputs in the neural network.
        """
        printString = "Fixed Inputs: { "
        for input_id, value in self.fixedInputs.items():
            printString = printString + str(f"\nFixed Input ID:{input_id}, Value:{value}")
        printString = printString + "\n}\n"
        print(printString)
        return printString
            
    def printPerceptronValues(self):
        """
        Prints and returns a string representation of the current values of all perceptrons in the neural network.

        This function iterates over each perceptron in the network, retrieves its current value from the last forward output, 
        and appends a string representation of the perceptron's ID and value to the output string.

        Returns:
            str: A string representation of the current values of all perceptrons in the neural network.
        """
        if len(self.forwardOutputs) == 0:
            return "Can't provide perceptron values, no forward output yet.\n"
        printString = "Perceptron Values: {"
        for perceptron_id, perceptron in self.perceptrons.items():
            printString = printString + str(f"\nPerceptron ID:{perceptron_id}, Value:{self.forwardOutputs[len(self.forwardOutputs)-1][perceptron_id]}")
        printString = printString + "\n}\n"
        print(printString)
        return printString
    
    def printLastResults(self):
        """
        Prints the last results of the neural network.

        This function retrieves the last forward output of the neural network and prints a string representation of the results.
        The string includes the ID and value of each perceptron in the network.

        Returns:
            str: A string representation of the last results of the neural network.
        """
        if len(self.forwardOutputs) == 0:
            return "Can't provide last results, no forward output yet.\n"
        last = self.forwardOutputs[len(self.forwardOutputs)-1];
        printString = "Last Results: {"
        for perceptron_id, value in last.items():
            printString = printString + str(f"\nPerceptron ID:{perceptron_id}, Value:{value}")
        printString = printString + "\n}\n"
        print(printString)
        return printString
            
    def printPerceptronFixedInputs(self,perceptronID):
        """
        Prints a string representation of the fixed inputs of a specific perceptron in the network.

        Parameters:
        perceptronID (int): The ID of the perceptron whose fixed inputs are to be printed.

        Returns:
        str: A string representation of the fixed inputs of the specified perceptron.
        """
        input_ids = self.getFixedInputs(perceptronID)
        printString = "\nFixed Inputs of Perceptron: " + str(perceptronID) + " are: { "
        for input_id in input_ids:
            printString = printString + str(f"\nInput ID:{input_id}, Value:{self.fixedInputs[input_id]}")
        printString = printString + "\n}\n"
        print(printString)
        return printString
    
    def printPerceptronInputs(self,perceptronID):
        """
        Prints a string representation of the inputs of a specific perceptron in the network.

        Parameters:
        perceptronID (int): The ID of the perceptron whose inputs are to be printed.

        Returns:
        str: A string representation of the inputs of the specified perceptron.
        """
        input_ids = self.getInputs(perceptronID)
        printString = "\n Inputs of Perceptron: " + str(perceptronID) + " are: { "
        for input_id in input_ids:
            printString = printString + str(f"\n Input ID:{input_id}, Value:{self.getInputValue(input_id)}")
        printString = printString + "\n}\n"
        print(printString)
        return printString
        
    def printCurrentPerceptronWeights(self,perceptronID):
        """
        Prints a string representation of the current weights of a specific perceptron in the network.

        Parameters:
        perceptronID (int): The ID of the perceptron whose weights are to be printed.

        Returns:
        str: A string containing the current weights of the specified perceptron.
        """
        printString = f"\nCurrent Weights of Perceptron (ID: {perceptronID})  are: "
        printString = printString + "{ "
        for weight in self.perceptrons[perceptronID].getWeights():
            printString = printString + str(f"\nWeight: {weight}, ")
        printString = printString + "\n}\n"
        print(printString)
        return printString
        
    def printAllWeights(self):
        """
        Prints a string representation of all perceptron weights in the network.

        Returns:
        str: A string containing all perceptron weights.
        """
        printString = "All Weights: {"
        for perceptron_id, perceptron in self.perceptrons.items():
            printString = printString + str(f"\nPerceptron ID:{perceptron_id}, Weights:{perceptron.getWeights()}")
        printString = printString + "\n}\n"
        print(printString)
        return printString
        
            
    def printPerceptronInConnections(self,perceptronID):
        """
        Prints the input connections of a perceptron with the given ID.

        Parameters:
        perceptronID (int): The ID of the perceptron.

        Returns:
        str: A string representation of the input connections.
        """
        input_ids = self.getInputs(perceptronID)
        printString = "Inputs of Perceptron: " + str(perceptronID) + " are: { "
        for input_id in input_ids:
            printString = printString + str(input_id) + " "
        printString = printString + "\n}\n"
        print(printString)
        return printString
            
    def printPerceptronOutConnections(self,perceptronID):
        """
        Prints the output connections of a perceptron with the given ID.

        Parameters:
        perceptronID (int): The ID of the perceptron.

        Returns:
        str: A string representation of the output connections.
        """
        output_ids = self.getOutputs(perceptronID)
        printString = "Outputs of Perceptron: " + str(perceptronID) + " are: { "
        for output_id in output_ids:
            printString = printString + f"\nPerceptron (ID: {perceptronID}) Outputs to Perceptron (ID:{output_id}), Value:{self.getPerceptronOutputValue(output_id)}"
        printString = printString + "\n}\n"
        print(printString)
        return printString
    
    def printPercentronConnections(self):
        printString = "Connections: {"
        for out_id,in_ids in self.outConnections.items():
            for in_id in in_ids:
                printString = printString + str(f"\nPerceptron (ID: {out_id}) Outputs to Perceptron (ID:{in_id}), ")
                if len(self.forwardOutputs) > 0:
                    printString = printString + f"Value:{self.getPerceptronOutputValue(in_id)}"
        printString = printString + "\n}\n"
        print(printString)
        return printString
            
    
    def dumpCurrentNetworkInfoToFile(self, filename = "netinfo.txt"):
        """
        Dumps the current state of the neural network to a file.

        Parameters:
        filename (str): The name of the file to write the network information to.

        Note:
        The filename parameter is not used in this implementation. The network information is always written to a file named "netinfo.txt".

        Returns:
        None
        """
        file = open(filename, "w")
        file.write(self.printFixedInputs())
        file.write(self.printPerceptronValues())
        file.write(self.printLastResults())
        file.write(self.printAllWeights())
        file.write(self.printPercentronConnections())
        file.close()
            
        