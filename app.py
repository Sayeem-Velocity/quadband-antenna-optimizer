from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('rf_ht_model.joblib')

# Homepage Route
@app.route('/')
def home():
    return render_template('index.html')

# Optimizer Page Route (Form & Prediction)
@app.route('/optimizer', methods=['GET', 'POST'])
def optimizer():
    result = None
    if request.method == 'POST':
        try:
            # Extract inputs
            freq = float(request.form['frequency'])             # Frequency (GHz)
            patch_len = float(request.form['patch_length'])     # Patch length (mm)
            patch_wid = float(request.form['patch_width'])      # Patch width (mm)
            substr_len = float(request.form['substrate_length'])# Substrate length (mm)
            substr_wid = float(request.form['substrate_width'])# Substrate width (mm)

            # Create feature array
            features = np.array([[freq, patch_len, patch_wid, substr_len, substr_wid]])

            # Predict using the loaded model
            s11_pred = model.predict(features)[0]

            # Round and store result
            result = {'s11_db': f"{s11_pred:.2f}"}

        except Exception as e:
            result = {'s11_db': f"Error: {str(e)}"}

    return render_template('optimizer.html', result=result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
