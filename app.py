from flask import Flask, request, render_template, jsonify
import numpy as np
from nn import TicTacToeNN

app = Flask(__name__)

# Initialize Neural Network
nn = TicTacToeNN()
try:
    nn.load_model("data/model.npy")
except FileNotFoundError:
    print("Model not found. Starting with random weights.")

# Game logic
def check_winner(board):
    win_positions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
        [0, 4, 8], [2, 4, 6]              # diagonals
    ]
    for pos in win_positions:
        if abs(board[pos[0]] + board[pos[1]] + board[pos[2]]) == 3:
            return True
    return False

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/move", methods=["POST"])
def move():
    data = request.get_json()
    board = np.array(data["board"], dtype=int)

    # Player's move
    board[data["player_move"]] = 1

    if check_winner(board):
        return jsonify({"status": "win", "board": board.tolist()})

    # AI's move
    ai_move = nn.predict_move(board)
    board[ai_move] = -1

    if check_winner(board):
        return jsonify({"status": "lose", "board": board.tolist()})

    if 0 not in board:
        return jsonify({"status": "draw", "board": board.tolist()})

    return jsonify({"status": "continue", "board": board.tolist()})

if __name__ == "__main__":
    app.run(debug=True,port=8000)
