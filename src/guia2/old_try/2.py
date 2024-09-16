import guia2.old_try.Network as nn
import numpy as np

def sigmoid(z):
        return 1 / (1 + np.exp(-z))

def easy_network_test():
    net = nn.NeuronalNetwork()
    
    net.disableBias()

    # Adding perceptrons and connections
    perceptron1_id = net.addPerceptron()
    perceptron2_id = net.addPerceptron()
    perceptron3_id = net.addPerceptron()

    # Add fixed inputs
    net.addFixedInput(id=0)
    net.addFixedInput(id=1)
    
    # Connect fixed inputs to perceptrons
    net.connectFixedInput(fixedInputId=1, perceptronId=perceptron1_id)
    net.connectFixedInput(fixedInputId=0, perceptronId=perceptron1_id)
    net.connectFixedInput(fixedInputId=1, perceptronId=perceptron2_id)
    net.connectFixedInput(fixedInputId=0, perceptronId=perceptron2_id)

    # Add connections between perceptrons
    net.connectPerceptrons(outputPerceptronIDID=perceptron1_id, inputPerceptronIDID=perceptron3_id)
    net.connectPerceptrons(outputPerceptronIDID=perceptron2_id, inputPerceptronIDID=perceptron3_id)

    # Update the layer structure
    net.updateLayerStructure()

    # Plot the network
    net.plot_network()
    
def complex_network_test():
    net = nn.Network()

    # Adding perceptrons
    perceptron_ids = [net.addPerceptron() for _ in range(15)]  # Assume 5 inputs per perceptron

    # Add fixed inputs
    for i in range(4):
        net.addFixedInput(i)

    # Connect fixed inputs to multiple first-layer perceptrons
    net.connectFixedInput(fixedInputId=0, perceptronId=perceptron_ids[0])
    net.connectFixedInput(fixedInputId=0, perceptronId=perceptron_ids[1])
    net.connectFixedInput(fixedInputId=1, perceptronId=perceptron_ids[2])
    net.connectFixedInput(fixedInputId=1, perceptronId=perceptron_ids[3])
    net.connectFixedInput(fixedInputId=2, perceptronId=perceptron_ids[4])
    net.connectFixedInput(fixedInputId=2, perceptronId=perceptron_ids[5])
    net.connectFixedInput(fixedInputId=3, perceptronId=perceptron_ids[6])
    net.connectFixedInput(fixedInputId=3, perceptronId=perceptron_ids[7])

    # Add complex connections between first layer and second layer perceptrons
    net.connectPerceptrons(outputPerceptronID=perceptron_ids[0], inputPerceptronID=perceptron_ids[8])
    net.connectPerceptrons(outputPerceptronID=perceptron_ids[1], inputPerceptronID=perceptron_ids[8])
    net.connectPerceptrons(outputPerceptronID=perceptron_ids[2], inputPerceptronID=perceptron_ids[9])
    net.connectPerceptrons(outputPerceptronID=perceptron_ids[3], inputPerceptronID=perceptron_ids[9])
    net.connectPerceptrons(outputPerceptronID=perceptron_ids[4], inputPerceptronID=perceptron_ids[10])
    net.connectPerceptrons(outputPerceptronID=perceptron_ids[5], inputPerceptronID=perceptron_ids[10])
    net.connectPerceptrons(outputPerceptronID=perceptron_ids[6], inputPerceptronID=perceptron_ids[11])
    net.connectPerceptrons(outputPerceptronID=perceptron_ids[7], inputPerceptronID=perceptron_ids[11])

    # Connecting second layer perceptrons to third layer perceptrons
    net.connectPerceptrons(outputPerceptronID=perceptron_ids[8], inputPerceptronID=perceptron_ids[12])
    net.connectPerceptrons(outputPerceptronID=perceptron_ids[9], inputPerceptronID=perceptron_ids[12])
    net.connectPerceptrons(outputPerceptronID=perceptron_ids[10], inputPerceptronID=perceptron_ids[13])
    net.connectPerceptrons(outputPerceptronID=perceptron_ids[11], inputPerceptronID=perceptron_ids[13])

    # Connecting third layer perceptrons to the final perceptron
    net.connectPerceptrons(outputPerceptronID=perceptron_ids[12], inputPerceptronID=perceptron_ids[14])
    net.connectPerceptrons(outputPerceptronID=perceptron_ids[13], inputPerceptronID=perceptron_ids[14])

    # Update the layer structure
    net.updateLayerStructure()

    # Plot the network if plotting is enabled
    if net.enableGraph:
        net.plot_network()
    net.dumpCurrentNetworkInfoToFile()

def test_forward():
    # Initialize the neural network
    net = nn.Network()
    
    net.disableAutomaticBiases()
    
    # Add perceptrons to the network
    net.addPerceptron(0)  # Input layer perceptron
    net.addPerceptron(1)  # Hidden layer perceptron
    net.addPerceptron(2)  # Output layer perceptron

    # Add fixed input to perceptron 0
    net.addFixedInput(0) 
    net.setInputValue(0, 1.0) # Set fixed input to 1.0 

    # Connect fixed input 0 to perceptron 0
    net.connectFixedInput(0, 0)
    net.connectPerceptrons(0, 1)
    net.connectPerceptrons(1, 2)


    # Set weights for perceptrons manually for testing
    net.setFixedToPerWeight(0, 0, 0.5)  # Perceptron 0 has one input
    net.setPerToPerWeight(1,0,0.4)  # Perceptron 1 has one input
    net.setPerToPerWeight(2,1,0.6)  # Perceptron 2 has one input
    
    # POST SETTING UP VALUES
    net.printAllWeights()
    net.printFixedInputs()

    # Update layer structure
    net.updateLayerStructure()

    # Run forward pass
    net.forward()  # Run forward pass

    # Get the outputs from the last layer
    output = net.getLastForwardOutputs()

    # Check the output values
    perceptrons_outs = []
    perceptrons_outs.append(output[0])
    perceptrons_outs.append(output[1])
    perceptrons_outs.append(output[2])
    
    print("Output of perceptron 0:", perceptrons_outs[0])
    print("Output of perceptron 1:", perceptrons_outs[1])
    print("Output of perceptron 2:", perceptrons_outs[2])

    # Expected forward pass values (based on perceptron weights and inputs)
    expected_output_0 = sigmoid(1.0 * 0.5)  # Perceptron 0 output
    expected_output_1 = sigmoid(expected_output_0 * 0.4)  # Perceptron 1 output
    expected_output_2 = sigmoid(expected_output_1 * 0.6)  # Perceptron 2 output

    assert np.isclose(perceptrons_outs[0], expected_output_0), f"Expected {expected_output_0}, got {perceptrons_outs[0]}"
    assert np.isclose(perceptrons_outs[1], expected_output_1), f"Expected {expected_output_1}, got {perceptrons_outs[1]}"
    assert np.isclose(perceptrons_outs[2], expected_output_2), f"Expected {expected_output_2}, got {perceptrons_outs[2]}"

    print("Forward propagation test passed.")
    net.dumpCurrentNetworkInfoToFile()

def test_backward():
    # Initialize the neural network
    net = nn.Network()
    
    net.disableAutomaticBiases()

    # Add perceptrons to the network
    net.addPerceptron(0)  # Input layer perceptron
    net.addPerceptron(1)  # Hidden layer perceptron
    net.addPerceptron(2)  # Output layer perceptron

    # Add fixed input to perceptron 0
    net.addFixedInput(0) 
    net.setInputValue(0, 1.0) # Set fixed input to 1.0 

    # Connect fixed input 0 to perceptron 0
    net.connectFixedInput(0, 0)
    net.connectPerceptrons(0, 1)
    net.connectPerceptrons(1, 2)

    # Set weights for perceptrons manually for testing
    net.setFixedToPerWeight(0, 0, 0.5)  # Perceptron 0 has one input
    net.setPerToPerWeight(1,0,0.4)  # Perceptron 1 has one input
    net.setPerToPerWeight(2,1,0.6)  # Perceptron 2 has one input

    # Update layer structure and run forward pass
    net.updateLayerStructure()
    
    net.plot_network()
    net.forward()  # Run forward pass

    # Define desired output (y_d) for backpropagation
    y_d = {2: 0.8}  # Desired output for perceptron 2 is 0.8
    
    # Run backward propagation
    delta = net.backward(y_d)

    # Check the computed deltas
    delta_2 = delta[2]  # Delta for perceptron 2 (output layer)
    delta_1 = delta[1]  # Delta for perceptron 1 (hidden layer)
    delta_0 = delta[0]  # Delta for perceptron 0 (input layer)

    # Expected delta values based on the forward pass and desired output
    expected_output_2 = net.getPerceptronOutputValue(2)
    expected_delta_2 = expected_output_2 - y_d[2]
    
    expected_output_1 = net.getPerceptronOutputValue(1)
    expected_delta_1 = expected_delta_2 * net.getConnectionWeigths(2, 1)
    
    expected_output_0 = net.getPerceptronOutputValue(0)
    expected_delta_0 = expected_delta_1 * net.getConnectionWeigths(1, 0)

    # Assertions to validate delta values
    assert np.isclose(delta_2, expected_delta_2), f"Expected delta for P2: {expected_delta_2}, got {delta_2}"
    assert np.isclose(delta_1, expected_delta_1), f"Expected delta for P1: {expected_delta_1}, got {delta_1}"
    assert np.isclose(delta_0, expected_delta_0), f"Expected delta for P0: {expected_delta_0}, got {delta_0}"

    print("Backward propagation test passed.")

# Helper function for sigmoid calculation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def test_random_network():
    # Initialize the neural network
    net = nn.NeuronalNetwork()
    
    nn.getFixedInputs()

    # Create some fixed inputs
    num_fixed_inputs = 3
    for i in range(num_fixed_inputs):
        net.addFixedInput(i)

    # Create some perceptrons
    num_perceptrons = 5
    for i in range(num_perceptrons):
        net.addPerceptron(i)

    # Connect fixed inputs to perceptrons randomly
    for i in range(num_fixed_inputs):
        perceptron_id = np.random.randint(0, num_perceptrons)
        net.connectFixedInput(i, perceptron_id)

    # Randomly connect perceptrons to each other
    for i in range(num_perceptrons):
        output_perceptron = i
        input_perceptron = np.random.randint(0, num_perceptrons)
        if output_perceptron != input_perceptron:
            net.connectPerceptrons(output_perceptron, input_perceptron)

    # Set random values to fixed inputs
    for i in range(num_fixed_inputs):
        random_value = np.random.rand()
        net.setInputValue(i, random_value)
        print(f"Fixed Input {i} value set to {random_value}")

    # Update the layer structure of the network before forward pass
    net.updateLayerStructure()

    # Perform forward propagation and print the outputs
    outputs = net.forward()
    print("\nOutputs from forward propagation:")
    for perceptron_id, output in outputs.items():
        print(f"Perceptron {perceptron_id}: {output}")
    
    net.plot_network()
    
def getError(y, y_d):
    acum = 0;
    for i in range(len(y)):
        if y[i] != y_d[i]:
            acum += 1
    return acum/len(y)
    
    
def real_test():
    # Read CSV file
    filename = 'XOR_trn.csv'
    data = np.genfromtxt(filename, delimiter=',')
    
    
    
    net = nn.Network()
    per = []
    inp = []

    # Add perceptrons
    per.append(net.addPerceptron(0))
    per.append(net.addPerceptron(1))
    per.append(net.addPerceptron(2))
    
    # Inputs
    inp.append(net.addFixedInput(0))
    inp.append(net.addFixedInput(1))
    
    # Connect perceptrons
    net.connectPerceptrons(0,2)
    net.connectPerceptrons(1,2)
    
    # Connect fixed inputs
    
    net.connectFixedInput(0, 0)
    net.connectFixedInput(1, 1)
    net.connectFixedInput(0, 1)
    net.connectFixedInput(1, 0)
    
    net.updateLayerStructure()
    
    net.plot_network()
    
    
    outputs = [];
    for i in range(len(data)):
        net.setInputValue(0, data[i][0])
        net.setInputValue(1, data[i][1])
        net.forward()
        
    outputs = net.getForwardsOutputs()
    print(outputs)
    print(getError(outputs, data[:,2]))
    
    
    
# Run the test
# test_backward()

# complex_network_test()
real_test()