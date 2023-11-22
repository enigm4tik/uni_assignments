# Machine Learning Basics
# Assignment #07: ANN - feed forward

from random import uniform


def forward_neural_network(inputs):
    """
    Generate a neural network with given inputs and random biases and weights
    Layer 1: 3 nodes
    Layer 2: 4 nodes
    Output: 1 node
    :param inputs: list of input values (int or float)
    :return: calculated output and list of weights and biases used in network
    """
    layer1, settings = calculate_network_layer(inputs, 3)
    layer2, settings2 = calculate_network_layer(layer1, 4)
    output, settings3 = calculate_network_layer(layer2, 1)
    return output[0], settings + settings2 + settings3


def calculate_network_layer(inputs, amount_of_nodes):
    """
    Calculate the layer for given inputs and amount of nodes.
    The weights and bias are created with a pseudo random generator.
    :param inputs: list of input values (int or float)
    :param amount_of_nodes: amount of nodes (int)
    :return: list of <amount of nodes> nodes, list of dictionary containing {nodes, weights, biases}
    """
    weights = [round(uniform(-3, 3), 2) for i in range(len(inputs) * amount_of_nodes)]
    biases = [round(uniform(-3, 3), 2) for i in range(amount_of_nodes)]
    nodes = []
    for i in range(amount_of_nodes):
        nodes.append(calculate_relu(weights[i * len(inputs):i * len(inputs) + len(inputs)], inputs, biases[i]))
    return nodes, [{"nodes": amount_of_nodes, "weights": weights, "biases": biases}]


def calculate_relu(weights, inputs, bias):
    """
    Calculate result for given inputs, weights and bias
    Apply ReLu on the result
    :param weights: list of weights for inputs
    :param inputs: list of input values
    :param bias: list of biases
    :return: calculated output or 0 if value < 0
    """
    calculated_value = 0
    for i in range(len(weights)):
        calculated_value += (weights[i] * inputs[i])
    calculated_value += bias
    if calculated_value > 0:
        return calculated_value
    else:
        return 0


def pretty_print_best_settings(iterations, inputs, target):
    """
    Print the best result after <iterations> iterations running through randomized neural network.
    :param iterations: amount of runs to get to the best network
    :param inputs: list of initial values (int or float)
    :param target: given expected target value
    """
    best_loss = 50000  # big number to find best
    best_result = 0
    best_settings = []
    for index in range(iterations):
        result, settings = forward_neural_network(inputs)
        loss = abs(result - target)
        if loss < best_loss:
            best_loss = loss
            best_settings = settings
            best_result = result
    print(f"{'':_^150}")
    print(
        f"After {iterations} iterations the best loss function had {round(best_loss, 2)} loss with result {round(best_result, 2)}.")
    print(f"{'':_^150}")
    print(f"{'Feed Forward Neural Network'}")
    print(f"{'':_^150}")
    print(f"{'Layer':^8} | {'Nodes':^5} | {'Bias':30} | {'Weights'} ")
    for index, setting in enumerate(best_settings):
        if not index == len(best_settings) - 1:
            print(
                f"{index + 1:^8} | {best_settings[index]['nodes']:^5} | {str(best_settings[index]['biases']):30} | {str(best_settings[index]['weights'])}")
        else:
            print(
                f"{'Output':^8} | {best_settings[index]['nodes']:^5} | {str(best_settings[index]['biases']):30} | {str(best_settings[index]['weights'])}")


input_values = [9, 3]
target_value = 7
pretty_print_best_settings(100, input_values, target_value)
