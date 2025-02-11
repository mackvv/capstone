<?php
/**
 * Plugin Name: SMILES Predictor
 * Description: A shortcode to predict SMILES strings using an external API.
 * Version: 1.0
 * Author: Mackenzie Van Vliet
 */

if (!defined('ABSPATH')) {
    exit; // Prevent direct access
}

function smiles_form_shortcode() {
    ob_start(); // Start output buffering
    ?>
    <form id="smiles-form">
        <label for="smiles-input">Enter SMILES String:</label>
        <input type="text" id="smiles-input" name="smiles" required>
        <button type="submit">Submit</button>
    </form>

    <div id="smiles-result"></div>

    <script>
        document.getElementById('smiles-form').onsubmit = async function(event) {
            event.preventDefault(); 

            var smiles = document.getElementById('smiles-input').value.trim();
            var resultDiv = document.getElementById('smiles-result');

            // Clear previous results
            resultDiv.innerHTML = '';

            // Check if input is empty
            if (!smiles) {
                resultDiv.innerHTML = `<p style="color: red;">Error: Please enter a SMILES string.</p>`;
                return;
            }

            // Show loading state
            resultDiv.innerHTML = `<p style="color: blue;">Processing...</p>`;

            try {
                let response = await fetch('https://capstonev1.azurewebsites.net/predict', { 
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 'smiles': smiles }),
                });

                let textData = await response.text(); // Read as text
                console.log("Raw API Response:", textData); // Log raw response

                let prediction;
                try {
                    let jsonData = JSON.parse(textData); // Try to parse as JSON
                    prediction = jsonData.data; // Extract prediction value
                } catch (error) {
                    console.warn("Response is not JSON, treating as plain text.");
                    prediction = textData.trim(); // Use raw text
                }

                // Map prediction to output
                let outputMessage;
                if (prediction === "Prediction: 0" || prediction === "0") {
                    outputMessage = `<p style="color: green;">Result: Non-Toxic</p>`;
                } else if (prediction === "Prediction: 1" || prediction === "1") {
                    outputMessage = `<p style="color: red;">Result: Toxic</p>`;
                } else {
                    outputMessage = `<p style="color: red;">Error: Unexpected response format.</p>`;
                }

                resultDiv.innerHTML = outputMessage;

            } catch (error) {
                console.error("Fetch error:", error);
                resultDiv.innerHTML = `<p style="color: red;">Error: ${error.message || 'Failed to fetch data'}</p>`;
            }
        };
    </script>
    <?php
    return ob_get_clean(); 
}
add_shortcode('smiles_form', 'smiles_form_shortcode');
?>
