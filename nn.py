import numpy as np

class TicTacToeNN:
    def __init__(self, input_size=9, hidden_size=16, output_size=9):
        # Initialize weights and biases
        self.weights1 = np.random.rand(input_size, hidden_size) - 0.5
        self.bias1 = np.random.rand(hidden_size) - 0.5
        self.weights2 = np.random.rand(hidden_size, output_size) - 0.5
        self.bias2 = np.random.rand(output_size) - 0.5

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=0)

    def forward(self, board):
        hidden = self.relu(np.dot(board, self.weights1) + self.bias1)
        output = self.softmax(np.dot(hidden, self.weights2) + self.bias2)
        return output

    def predict_move(self, board):
        probabilities = self.forward(board)
        valid_moves = np.where(board == 0)[0]
        probabilities = probabilities[valid_moves]
        best_move = valid_moves[np.argmax(probabilities)]
        return best_move

    def save_model(self, file_path):
        np.save(file_path, [self.weights1, self.bias1, self.weights2, self.bias2])

    def load_model(self, file_path):
        self.weights1, self.bias1, self.weights2, self.bias2 = np.load(file_path, allow_pickle=True)
