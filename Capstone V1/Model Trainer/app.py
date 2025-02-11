#Local host training and validation 
#Description: Runs model training and saves model.pth and vocab.pth 
#Version: 1.0
#Author: Mackenzie Van Vliet

from flask import Flask, request, jsonify, render_template
import torch
import os
from model_utils import SMILESClassifier, encode_smiles, vocab, predict_smiles, train_model, train_loader, val_loader
from rdkit import Chem

# Initialize Flask app
app = Flask("capstone")

# Define model parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = len(vocab)
embed_dim = 128
hidden_dim = 256
output_dim = 3  # Adjust based on dataset classes

# Initialize model
model = SMILESClassifier(vocab_size, embed_dim, hidden_dim, output_dim).to(device)

# Load model checkpoint or train if not found
checkpoint_path = "model.pth"
if os.path.exists(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)  # Allow flexibility in state_dict keys
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print("No trained model found. Training a new model...")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=2)
    torch.save(model.state_dict(), checkpoint_path)
    print("Model training complete and saved.")

@app.route('/')
def home():
    """Render the home page with the input form."""
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the ASSAY_OUTCOME for the given SMILES string.
    Expect SMILES string input in the request body.
    """
    smiles = request.form.get('smiles')
    error_message = None

    if not smiles:
        error_message = "SMILES string is required."
    elif not Chem.MolFromSmiles(smiles):
        error_message = "Invalid SMILES string. Please enter a valid SMILES format."

    if error_message:
        return render_template("index.html", error_message=error_message)

    try:
        # Perform prediction
        prediction, confidence = predict_smiles(smiles, model, vocab, device)

        # Output raw model prediction and confidence
        return render_template("index.html", smiles=smiles, prediction=f"{prediction}", confidence=f"{confidence:.2%}")
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=False, use_reloader=False)
