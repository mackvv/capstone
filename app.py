from flask import Flask, request, jsonify, render_template
import torch
from model_utils import SMILESClassifier, encode_smiles, vocab, predict_smiles_with_probabilities
from rdkit import Chem

# Initialize Flask app
app = Flask("capstone")

# Check for GPU and set device
device = torch.device("cuda")
print(f"Using device: {device}")

# Model parameters
vocab_size = len(vocab)
embed_dim = 128
hidden_dim = 256
output_dim = 3  # Adjust this to the actual number of classes in your dataset


# Load the pre-trained model checkpoint (no training involved)
try:
    checkpoint = torch.load("model.pth", map_location=device)
    model.load_state_dict(checkpoint, strict=False)  # Allow flexibility in state_dict keys
    model.to(device)
    model.eval()  # Set the model to evaluation mode
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
        # Perform prediction with probabilities
        prediction, probabilities = predict_smiles_with_probabilities(model, smiles, vocab, device)

        # Map prediction to a readable label
        class_labels = ["Inactive", "Active Agonist", "Active Antagonist"]
        prediction_label = class_labels[prediction]

        # Prepare probabilities as a dictionary for display
        probabilities_dict = {
            class_labels[i]: f"{prob:.2%}" for i, prob in enumerate(probabilities)
        }

        return render_template(
            "index.html",
            smiles=smiles,
            prediction=prediction_label,
            probabilities=probabilities_dict,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=False, use_reloader=False)
