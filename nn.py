import numpy as np

class TicTacToeNN:
    def __init__(self, input_size=9, hidden_size=18, output_size=9):
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
    # Save weights and biases as a dictionary
        model = {
            'weights1': self.weights1,
            'bias1': self.bias1,
            'weights2': self.weights2,
            'bias2': self.bias2
        }
        np.save(file_path, model)  # Save the dictionary to a file

    def load_model(self, file_path):
        # Load the model dictionary from the file
        model = np.load(file_path, allow_pickle=True).item()
        self.weights1 = model['weights1']
        self.bias1 = model['bias1']
        self.weights2 = model['weights2']
        self.bias2 = model['bias2']


# Helper functions for training
def one_hot_encode_move(board, move):
    """Returns a one-hot encoded vector for the move"""
    encoding = np.zeros(9)
    encoding[move] = 1
    return encoding

def train_network(network, data, labels, epochs=1000, learning_rate=0.01):
    for epoch in range(epochs):
        total_loss = 0
        for board, move in zip(data, labels):
            # Forward pass
            board_input = np.array(board).flatten()
            predicted_probs = network.forward(board_input)
            
            # Compute loss (Cross-entropy loss)
            loss = -np.log(predicted_probs[np.argmax(move)])  # Use true move's probability
            total_loss += loss
            
            # Backpropagation (gradient descent step)
            # Compute gradients (simplified version, you could use a more sophisticated method)
            output_error = predicted_probs - move
            
            # Calculate hidden layer error (use ReLU derivative on hidden layer activations)
            hidden_activation = network.relu(np.dot(board_input, network.weights1) + network.bias1)
            hidden_error = np.dot(output_error, network.weights2.T) * (hidden_activation > 0)  # ReLU derivative
            
            # Update weights and biases using gradient descent
            network.weights2 -= learning_rate * np.outer(hidden_activation, output_error)
            network.bias2 -= learning_rate * output_error
            network.weights1 -= learning_rate * np.outer(board_input, hidden_error)
            network.bias1 -= learning_rate * hidden_error
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(data)}")
    
    # Save the model after training
    
    network.save_model("tic_tac_toe_model.npy")
    print("Model saved to 'tic_tac_toe_model.npy'.")


# Example training data (board configurations and moves)
# Each board is a 9-dimensional vector (flattened 3x3 board)
# Labels are one-hot encoded moves (for simplicity, we use just a few examples)

# Example data (input boards and the corresponding "best" move)
data = [
    [0, 0, 0, 0, 0, 0, 0, 0, 1],  # Example board
    [1, 0, 0, 0, 1, 0, 0, 0, 0],  # Example board
    [0, 1, 0, 1, 0, 0, 0, 0, 0],  # Example board
]

# Example labels (one-hot encoded)
labels = [
    one_hot_encode_move([0, 0, 0, 0, 0, 0, 0, 0, 1], 8),
    one_hot_encode_move([1, 0, 0, 0, 1, 0, 0, 0, 0], 4),
    one_hot_encode_move([0, 1, 0, 1, 0, 0, 0, 0, 0], 3),
]

# Initialize the TicTacToeNN and train it
nn = TicTacToeNN()
train_network(nn, data, labels, epochs=1000, learning_rate=0.01)