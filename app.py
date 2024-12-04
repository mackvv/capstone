from flask import Flask, request, jsonify, render_template
import torch
from model_utils import SMILESClassifier, encode_smiles, vocab, predict_smiles
from rdkit import Chem

# Initialize Flask app
app = Flask("capstone")

# Load your trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = len(vocab)
embed_dim = 128
hidden_dim = 256
output_dim = 3  # Adjust this to the actual number of classes in your dataset
model = SMILESClassifier(vocab_size, embed_dim, hidden_dim, output_dim)

# Load model checkpoint
try:
    checkpoint = torch.load("model.pth", map_location=device)
    model.load_state_dict(checkpoint, strict=False)  # Allow flexibility in state_dict keys
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

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
    else:
        # Validate the SMILES string
        if not Chem.MolFromSmiles(smiles):
            error_message = "Invalid SMILES string. Please enter a valid SMILES format."

    if error_message:
        return render_template("index.html", error_message=error_message)

    try:
        # Perform prediction
        prediction = predict_smiles(model, smiles, vocab, device)
        
        # Map prediction to a readable label
        if prediction == 0:
            prediction_label = "Inactive"
        elif prediction == 1:
            prediction_label = "Active Agonist"
        elif prediction == 2:
            prediction_label = "Active Antagonist"
        else:
            prediction_label = "Unknown"
        
        return render_template("index.html", smiles=smiles, prediction=prediction_label)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=False, use_reloader=False)
