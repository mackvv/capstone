#Model training 
#Description: Model training file
#Version: 1.0
#Author: Mackenzie Van Vliet

import os
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from rdkit import Chem
from tqdm import tqdm
from datetime import datetime

# Directory for saved files
SAVE_DIR = "saved_models"
VOCAB_PATH = os.path.join(SAVE_DIR, "vocab.pth")
DATA_CACHE_DIR = "cached_data"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Function to check valid SMILES
def is_valid_smiles(smiles):
    return isinstance(smiles, str) and Chem.MolFromSmiles(smiles) is not None

# Load dataset
file_path = os.path.join("C:\\", "Users", "macke", "OneDrive", "Desktop", "Capstone", "tox21-ahr-p1.csv")
df = pd.read_csv(file_path)
df = df.dropna(subset=["SMILES"])
df = df[df["SMILES"].apply(is_valid_smiles)]

# Split dataset
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["ASSAY_OUTCOME"])
train_smiles, train_outcomes = train_df["SMILES"].tolist(), train_df["ASSAY_OUTCOME"].tolist()
val_smiles, val_outcomes = val_df["SMILES"].tolist(), val_df["ASSAY_OUTCOME"].tolist()

# Tokenization function
def tokenize_smiles(smiles):
    return list(smiles)

# Create and save vocabulary
def create_or_load_vocab(train_smiles):
    if os.path.exists(VOCAB_PATH):
        return torch.load(VOCAB_PATH)

    token_freq = Counter(token for smiles in train_smiles for token in tokenize_smiles(smiles))
    token_freq["<pad>"] = float('inf')
    vocab = {token: idx for idx, (token, _) in enumerate(token_freq.items())}
    vocab["<pad>"] = 0

    torch.save(vocab, VOCAB_PATH)
    print(f"Vocab saved at {VOCAB_PATH}")
    return vocab

# Load or create vocab
vocab = create_or_load_vocab(train_smiles)

# Encode SMILES function
def encode_smiles(smiles, vocab):
    return [vocab.get(token, vocab["<pad>"]) for token in tokenize_smiles(smiles)]

# Cache Encoded Data
train_encoded_path = os.path.join(DATA_CACHE_DIR, "train_encoded.pt")
val_encoded_path = os.path.join(DATA_CACHE_DIR, "val_encoded.pt")

if os.path.exists(train_encoded_path) and os.path.exists(val_encoded_path):
    train_encoded = torch.load(train_encoded_path)
    val_encoded = torch.load(val_encoded_path)
else:
    train_encoded = [encode_smiles(s, vocab) for s in train_smiles]
    val_encoded = [encode_smiles(s, vocab) for s in val_smiles]
    torch.save(train_encoded, train_encoded_path)
    torch.save(val_encoded, val_encoded_path)
    print(f"Encoded data saved at {DATA_CACHE_DIR}")

# Dataset class
class SMILESDataset(Dataset):
    def __init__(self, smiles_encoded, outcomes):
        self.smiles_encoded = smiles_encoded
        self.outcomes = outcomes

    def __len__(self):
        return len(self.smiles_encoded)

    def __getitem__(self, idx):
        return torch.tensor(self.smiles_encoded[idx], dtype=torch.long), torch.tensor(self.outcomes[idx], dtype=torch.long)

# Collate function for padding
def collate_fn(batch):
    smiles, outcomes = zip(*batch)
    padded_smiles = nn.utils.rnn.pad_sequence(smiles, batch_first=True, padding_value=vocab["<pad>"])
    return padded_smiles, torch.tensor(outcomes)

# DataLoader
train_dataset = SMILESDataset(train_encoded, train_outcomes)
val_dataset = SMILESDataset(val_encoded, val_outcomes)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Model Definition (Bidirectional LSTM)
class SMILESClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SMILESClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # Combine both directions
        return self.fc(hidden)

# Training Function with Checkpoints
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    model.to(device)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0

        for smiles, outcomes in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            smiles, outcomes = smiles.to(device), outcomes.to(device)
            optimizer.zero_grad()
            predictions = model(smiles)
            loss = criterion(predictions, outcomes)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for smiles, outcomes in val_loader:
                smiles, outcomes = smiles.to(device), outcomes.to(device)
                predictions = model(smiles)
                loss = criterion(predictions, outcomes)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")

        # Save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
            print(f"Best model saved at {SAVE_DIR}/best_model.pth")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

# Prediction Function
def predict_smiles(smiles, model, vocab, device):
    model.eval()
    encoded_smiles = encode_smiles(smiles, vocab)
    input_tensor = torch.tensor(encoded_smiles, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(input_tensor)
        probabilities = torch.softmax(predictions, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

    return predicted_class, confidence
