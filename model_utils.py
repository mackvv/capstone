import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from rdkit import Chem
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from sklearn.metrics import accuracy_score

def calculate_accuracy(predictions, targets):
    preds = torch.argmax(torch.softmax(predictions, dim=1), dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    accuracy = accuracy_score(targets, preds)
    return accuracy

# Path to dataset
file_path = os.path.join("C:\\", "Users", "macke", "OneDrive", "Desktop", "Capstone", "cleaned_data.csv")

# Load dataset
df = pd.read_csv(file_path)

# Split dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["ASSAY_OUTCOME"])
train_smiles = train_df["SMILES"].tolist()
train_outcomes = train_df["ASSAY_OUTCOME"].tolist()
val_smiles = val_df["SMILES"].tolist()
val_outcomes = val_df["ASSAY_OUTCOME"].tolist()

# Tokenize SMILES
def tokenize_smiles(smiles):
    return list(smiles)

# Count token frequencies
token_freq = Counter(token for smiles in train_smiles for token in tokenize_smiles(smiles))
token_freq["<pad>"] = float('inf')  # Special token

# Create vocabulary
vocab = {token: idx for idx, (token, _) in enumerate(token_freq.items())}
vocab["<pad>"] = 0  # Pad token index

# Encode SMILES
def encode_smiles(smiles, vocab):
    return [vocab.get(token, vocab["<pad>"]) for token in tokenize_smiles(smiles)]

train_encoded = [encode_smiles(s, vocab) for s in train_smiles]
val_encoded = [encode_smiles(s, vocab) for s in val_smiles]

# Dataset and DataLoader
class SMILESDataset(Dataset):
    def __init__(self, smiles_encoded, outcomes):
        self.smiles_encoded = smiles_encoded
        self.outcomes = outcomes

    def __len__(self):
        return len(self.smiles_encoded)

    def __getitem__(self, idx):
        return torch.tensor(self.smiles_encoded[idx], dtype=torch.long), torch.tensor(self.outcomes[idx], dtype=torch.long)

def collate_fn(batch):
    smiles, outcomes = zip(*batch)
    padded_smiles = nn.utils.rnn.pad_sequence(smiles, batch_first=True, padding_value=vocab["<pad>"])
    return padded_smiles, torch.tensor(outcomes)

train_dataset = SMILESDataset(train_encoded, train_outcomes)
val_dataset = SMILESDataset(val_encoded, val_outcomes)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Model Definition
class SMILESClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SMILESClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])

# Metrics calculation
def calculate_metrics(predictions, targets):
    preds = torch.argmax(torch.softmax(predictions, dim=1), dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    precision = precision_score(targets, preds, average="weighted", zero_division=0)
    recall = recall_score(targets, preds, average="weighted", zero_division=0)
    f1 = f1_score(targets, preds, average="weighted", zero_division=0)
    return precision, recall, f1

# Training loop with logging and visualization
# Training loop with accuracy calculation
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=11):
    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = f"runs/{timestamp}_epochs_{epochs}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for smiles, outcomes in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", unit="batch"):
            smiles, outcomes = smiles.to(device), outcomes.to(device)
            optimizer.zero_grad()
            predictions = model(smiles)
            loss = criterion(predictions, outcomes)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        train_loss = epoch_train_loss / len(train_loader)
        writer.add_scalar("Loss/Train", train_loss, epoch)

        model.eval()
        epoch_val_loss = 0
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for smiles, outcomes in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation", unit="batch"):
                smiles, outcomes = smiles.to(device), outcomes.to(device)
                predictions = model(smiles)
                loss = criterion(predictions, outcomes)
                epoch_val_loss += loss.item()
                
                # Collect predictions and targets for accuracy calculation
                all_predictions.append(predictions)
                all_targets.append(outcomes)

        val_loss = epoch_val_loss / len(val_loader)
        writer.add_scalar("Loss/Validation", val_loss, epoch)

        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        accuracy = calculate_accuracy(all_predictions, all_targets)
        writer.add_scalar("Accuracy/Validation", accuracy, epoch)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}")

    writer.close()


# Prediction Function with Probabilities
def predict_smiles_with_probabilities(model, smiles, vocab, device):
    model.eval()
    encoded = encode_smiles(smiles, vocab)
    input_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
    return prediction, probabilities.cpu().numpy().flatten()

# Main execution
device = torch.device("cuda")
vocab_size = len(vocab)
embed_dim = 128
hidden_dim = 256
output_dim = 3
model = SMILESClassifier(vocab_size, embed_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50)
