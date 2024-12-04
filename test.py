import requests
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define the URL of the Flask API endpoint
url = 'http://127.0.0.1:5000/predict'  # Change this to your actual server address

# Load the SMILES strings from the cleaned_data.csv file
df = pd.read_csv('cleaned_data.csv')  # Ensure the 'cleaned_data.csv' file is in the correct location

# Extract the SMILES strings from the 'SMILES' column
smiles_strings = df['SMILES'].tolist()


smiles_strings = smiles_strings[:1000]

# Function to send the request and handle the response
def send_request(smiles):
    payload = {'smiles': smiles}
    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            return response.json()  # Return the JSON response
        else:
            return {"error": f"Request failed with status code {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# Initialize a list to store the results
results = []

# Use ThreadPoolExecutor to send requests concurrently
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = []
    
    for smiles in smiles_strings:  # Iterate over the first 10 SMILES strings
        futures.append(executor.submit(send_request, smiles))  # Submit request to executor
    
    # Collect the results as they are completed
    for future in as_completed(futures):
        results.append(future.result())

# Save the results to a file
with open('predictions_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Predictions saved to 'predictions_results.json'")
