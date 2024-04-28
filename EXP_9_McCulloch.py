

# ***McCulloch Pitts Model***


import numpy as np

class McCullochPittsNeuron:
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.weights = np.random.rand(num_inputs)
        self.threshold = np.random.rand()

    def activate(self, inputs):
        weighted_sum = np.dot(inputs, self.weights)
        if weighted_sum >= self.threshold:
            return 1
        else:
            return 0

    def train_perceptron(self, inputs, target_output, learning_rate=0.1, max_epochs=100):
        for epoch in range(max_epochs):
            error_count = 0
            for input_data, target in zip(inputs, target_output):
                prediction = self.activate(input_data)
                error = target - prediction
                if error != 0:
                    error_count += 1
                    self.weights += learning_rate * error * input_data
                    self.threshold -= learning_rate * error
            if error_count == 0:
                print(f"\nPerceptron converged in {epoch+1} epochs.")
                break
        else:
            print("\nPerceptron did not converge.")

# Functions to create inputs and target outputs for logical functions
def generate_logical_data(num_inputs):
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    if num_inputs == 1:
        inputs = inputs[:, 0].reshape(-1, 1)
    targets = np.array([0, 0, 0, 1])
    return inputs, targets

# Logical functions
def logical_and():
    inputs, targets = generate_logical_data(2)
    neuron = McCullochPittsNeuron(num_inputs=2)
    neuron.train_perceptron(inputs, targets)
    return neuron

def logical_or():
    inputs, targets = generate_logical_data(2)
    targets = np.logical_not(targets).astype(int)  # For OR, we invert the targets for training NOT
    neuron = McCullochPittsNeuron(num_inputs=2)
    neuron.train_perceptron(inputs, targets)
    return neuron

def logical_not():
    inputs, targets = generate_logical_data(1)
    targets = np.logical_not(targets).astype(int)  # For NOT, we invert the targets for training NOT
    neuron = McCullochPittsNeuron(num_inputs=1)
    neuron.train_perceptron(inputs, targets)
    return neuron

def logical_nand():
    inputs, targets = generate_logical_data(2)
    targets = np.logical_not(targets).astype(int)  # For NAND, we invert the targets for training NOT
    neuron = McCullochPittsNeuron(num_inputs=2)
    neuron.train_perceptron(inputs, targets)
    return neuron

def logical_nor():
    inputs, targets = generate_logical_data(2)
    neuron = McCullochPittsNeuron(num_inputs=2)
    neuron.train_perceptron(inputs, targets)
    return neuron

# Test the logical functions
if __name__ == "__main__":
    print("\nLogical AND:")
    and_neuron = logical_and()
    print("\nTesting the trained AND neuron:")
    for input_data in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        print(f"\nInput: {input_data},\t Output: {and_neuron.activate(input_data)}")

    print("\n------------------------------")
    print("\nLogical OR:")
    or_neuron = logical_or()
    print("\nTesting the trained OR neuron:")
    for input_data in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        print(f"\nInput: {input_data},\tOutput: {or_neuron.activate(input_data)}")

    print("\n------------------------------")
    print("\nLogical NOT:")
    not_neuron = logical_not()
    print("\nTesting the trained NOT neuron:")
    for input_data in [0, 1]:
        print(f"\nInput: {input_data},\t Output: {not_neuron.activate([input_data])}")

    print("\n------------------------------")
    print("\nLogical NAND:")
    nand_neuron = logical_nand()
    print("\nTesting the trained NAND neuron:")
    for input_data in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        print(f"\nInput: {input_data}, \tOutput: {nand_neuron.activate(input_data)}")

    print("\n-------------------------------")
    print("\nLogical NOR:")
    nor_neuron = logical_nor()
    print("\nTesting the trained NOR neuron:")
    for input_data in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        print(f"\nInput: {input_data},\t Output: {nor_neuron.activate(input_data)}")