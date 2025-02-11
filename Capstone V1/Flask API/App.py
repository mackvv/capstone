#Local host api testing
#Description: Flask app that loads trained model and vocab
#Version: 1.0
#Author: Mackenzie Van Vliet

import torch
import torch.nn as nn
import torch.nn.functional as F
from flask_cors import CORS
CORS(app)  # Allow all domains

from flask import Flask, request, jsonify

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

# Load vocab and model
vocab = torch.load(r"C:\Users\macke\OneDrive\Desktop\Flask\vocab.pth")
vocab_size = len(vocab)

model = SMILESClassifier(vocab_size=vocab_size, embed_dim=128, hidden_dim=256, output_dim=3)

# Load the model checkpoint
checkpoint_path = r"C:\Users\macke\OneDrive\Desktop\Flask\model.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Check if vocab size matches
old_vocab_size = checkpoint['embedding.weight'].size(0)
if old_vocab_size != vocab_size:
    # Adjust embedding weights to match vocab size
    print(f"Adjusting embedding layer from {old_vocab_size} to {vocab_size}")
    new_weights = F.pad(checkpoint['embedding.weight'], (0, 0, 0, vocab_size - old_vocab_size))
    checkpoint['embedding.weight'] = new_weights

model.load_state_dict(checkpoint)

# Initialize Flask app
app = Flask(__name__)

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    smiles = data.get('smiles')

    if not smiles:
        return jsonify({'error': 'No SMILES string provided'}), 400

    # Tokenize the SMILES string based on vocab
    tokens = [vocab.get(char, vocab.get('<UNK>')) for char in smiles]
    input_tensor = torch.tensor([tokens], dtype=torch.long)

    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()

    return jsonify({'prediction': prediction})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

