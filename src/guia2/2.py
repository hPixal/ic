import NeuronalNetwork as nn
import numpy as np

def sigmoid(z):
        return 1 / (1 + np.exp(-z))

def easy_network_test():
    net = nn.NeuronalNetwork()

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
    net.addPerceptronConnection(outputPerceptronIDID=perceptron1_id, inputPerceptronIDID=perceptron3_id)
    net.addPerceptronConnection(outputPerceptronIDID=perceptron2_id, inputPerceptronIDID=perceptron3_id)

    # Update the layer structure
    net.updateLayerStructure()

    # Plot the network
    net.plot_network()
    
def complex_network_test():
    net = nn.NeuronalNetwork()

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
    net.addPerceptronConnection(outputPerceptronID=perceptron_ids[0], inputPerceptronID=perceptron_ids[8])
    net.addPerceptronConnection(outputPerceptronID=perceptron_ids[1], inputPerceptronID=perceptron_ids[8])
    net.addPerceptronConnection(outputPerceptronID=perceptron_ids[2], inputPerceptronID=perceptron_ids[9])
    net.addPerceptronConnection(outputPerceptronID=perceptron_ids[3], inputPerceptronID=perceptron_ids[9])
    net.addPerceptronConnection(outputPerceptronID=perceptron_ids[4], inputPerceptronID=perceptron_ids[10])
    net.addPerceptronConnection(outputPerceptronID=perceptron_ids[5], inputPerceptronID=perceptron_ids[10])
    net.addPerceptronConnection(outputPerceptronID=perceptron_ids[6], inputPerceptronID=perceptron_ids[11])
    net.addPerceptronConnection(outputPerceptronID=perceptron_ids[7], inputPerceptronID=perceptron_ids[11])

    # Connecting second layer perceptrons to third layer perceptrons
    net.addPerceptronConnection(outputPerceptronID=perceptron_ids[8], inputPerceptronID=perceptron_ids[12])
    net.addPerceptronConnection(outputPerceptronID=perceptron_ids[9], inputPerceptronID=perceptron_ids[12])
    net.addPerceptronConnection(outputPerceptronID=perceptron_ids[10], inputPerceptronID=perceptron_ids[13])
    net.addPerceptronConnection(outputPerceptronID=perceptron_ids[11], inputPerceptronID=perceptron_ids[13])

    # Connecting third layer perceptrons to the final perceptron
    net.addPerceptronConnection(outputPerceptronID=perceptron_ids[12], inputPerceptronID=perceptron_ids[14])
    net.addPerceptronConnection(outputPerceptronID=perceptron_ids[13], inputPerceptronID=perceptron_ids[14])

    # Update the layer structure
    net.updateLayerStructure()

    # Plot the network if plotting is enabled
    if net.enableGraph:
        net.plot_network()
    net.dumpCurrentNetworkInfoToFile()

def test_forward():
    # Initialize the neural network
    net = nn.NeuronalNetwork()
    
    net.disableAutomaticBiases()

    # Add perceptrons to the network
    net.addPerceptron(0)  # Input layer perceptron
    net.addPerceptron(1)  # Hidden layer perceptron
    net.addPerceptron(2)  # Output layer perceptron

    # Add fixed input to perceptron 0
    net.addFixedInput(0)
    net.setInputValue(0, 1.0)  # Set fixed input to 1.0

    # Connect fixed input 0 to perceptron 0
    net.connectFixedInput(0, 0)

    # Connect perceptrons 0 -> 1, 1 -> 2
    net.addPerceptronConnection(0, 1)
    net.addPerceptronConnection(1, 2)

    # Set weights for perceptrons manually for testing
    net.setPerceptronWeights(0,[0.5])  # Perceptron 0 has one input
    net.setPerceptronWeights(1,[0.4])  # Perceptron 1 has one input
    net.setPerceptronWeights(2,[0.6])  # Perceptron 2 has one input
    
    # POST SETTING UP VALUES
    net.printAllWeights()
    net.printFixedInputs()

    
    net.updateLayerStructure()
    #net.plot_network()
    # Run forward pass
    output = net.forward()

    # Check the output values
    perceptrons_outs = []
    perceptrons_outs.append(net.perceptrons[0].forward())
    perceptrons_outs.append(net.perceptrons[1].forward())
    perceptrons_outs.append(net.perceptrons[2].forward())
    
    print("Output of perceptron 0:", perceptrons_outs[0])
    print("Output of perceptron 1:", perceptrons_outs[1])
    print("Output of perceptron 2:", perceptrons_outs[2])

    # Expected forward pass values (based on perceptron weights and inputs)
    expected_output_0 = sigmoid( 1.0 * 0.5 )  # Perceptron 0 output
    expected_output_1 = sigmoid(expected_output_0 * 0.4)  # Perceptron 1 output
    expected_output_2 = sigmoid(expected_output_1 * 0.6)  # Perceptron 2 output

    assert np.isclose(output[0], expected_output_0), f"Expected {expected_output_0}, got {output[0]}"
    assert np.isclose(output[1], expected_output_1), f"Expected {expected_output_1}, got {output[1]}"
    assert np.isclose(output[2], expected_output_2), f"Expected {expected_output_2}, got {output[2]}"

    print("Forward propagation test passed.")
    net.dumpCurrentNetworkInfoToFile()

import numpy as np

def test_random_network():
    # Initialize the neural network
    net = nn.NeuronalNetwork()

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
            net.addPerceptronConnection(output_perceptron, input_perceptron)

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

# Run the test
test_forward()

# complex_network_test()
