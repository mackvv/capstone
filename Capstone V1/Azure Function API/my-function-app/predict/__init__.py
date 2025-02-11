#Author: Mackenzie Van Vliet

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS
import azure.functions as func

# Get the absolute path of the function's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths for model and vocab
VOCAB_PATH = os.path.join(BASE_DIR, "vocab.pth")
MODEL_PATH = os.path.join(BASE_DIR, "model.pth")

# Ensure files exist before loading
if not os.path.exists(VOCAB_PATH):
    raise RuntimeError(f"File not found: {VOCAB_PATH}")
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"File not found: {MODEL_PATH}")

# Load vocab
vocab = torch.load(VOCAB_PATH)
vocab_size = len(vocab)

# Define the SMILESClassifier model
class SMILESClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SMILESClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        out = self.fc(lstm_out[:, -1, :])
        return out

# Initialize and load the model
model = SMILESClassifier(vocab_size=vocab_size, embed_dim=128, hidden_dim=256, output_dim=3)

# Load the model checkpoint
checkpoint = torch.load(MODEL_PATH, map_location="cpu")

# Adjust embedding weights if vocab size has changed
old_vocab_size = checkpoint['embedding.weight'].size(0)
if old_vocab_size != vocab_size:
    print(f"Adjusting embedding layer from {old_vocab_size} to {vocab_size}")
    new_weights = F.pad(checkpoint['embedding.weight'], (0, 0, 0, vocab_size - old_vocab_size))
    checkpoint['embedding.weight'] = new_weights

model.load_state_dict(checkpoint)
model.eval()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route('/predict', methods=['POST'])
def predict():
    """Handles SMILES prediction requests."""
    data = request.get_json()
    smiles = data.get('smiles')

    if not smiles:
        return jsonify({'error': 'No SMILES string provided'}), 400

    # Tokenize the SMILES string
    tokens = [vocab.get(char, vocab.get('<UNK>')) for char in smiles]
    input_tensor = torch.tensor([tokens], dtype=torch.long)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()

    return jsonify({'prediction': prediction})

# Azure function entry point
def main(req: func.HttpRequest) -> func.HttpResponse:
    """Azure Functions entry point."""
    try:
        # Get JSON from request
        req_body = req.get_json()
        smiles = req_body.get('smiles')

        if not smiles:
            return func.HttpResponse("Error: No SMILES string provided", status_code=400)

        # Tokenize the SMILES string
        tokens = [vocab.get(char, vocab.get('<UNK>')) for char in smiles]
        input_tensor = torch.tensor([tokens], dtype=torch.long)

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()

        return func.HttpResponse(f"Prediction: {prediction}", status_code=200)

    except Exception as e:
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)
