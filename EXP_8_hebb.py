

# ***Hebbian Learning Rule***

class HebbianNetwork:
    def __init__(self, input_size, output_size, learning_rate=1):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weights = [[0.0] * self.output_size for _ in range(self.input_size)]

    def train(self, input_data):
        for i in range(self.input_size):
            for j in range(self.output_size):
                self.weights[i][j] += self.learning_rate * input_data[i] * input_data[j]

    def predict(self, input_data):
        output = [0.0] * self.output_size
        for i in range(self.input_size):
            for j in range(self.output_size):
                output[j] += input_data[i] * self.weights[i][j]
        return output

"""***Example Training and Testing***"""

if __name__ == "__main__":
    network = HebbianNetwork(input_size=3, output_size=3)

    training_data = [
        [-1, 0, 1],
        [0.5, -1.5, 2.5],
        [0, 1, 2]
    ]

    # Online learning: train the network incrementally
    for data in training_data:
        network.train(data)
        print("\nWeights after training with data:", data)
        print(network.weights)

    # Test the network with a new input pattern
    test_input = [1, 0, 1]
    predicted_output = network.predict(test_input)
    print("\nPredicted output:", predicted_output)