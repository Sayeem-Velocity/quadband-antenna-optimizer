from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your pre-trained RF-HT model (predicts S11 from five geometry parameters)
model = joblib.load('rf_ht_model.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        # 1. Extract user inputs from form
        freq          = float(request.form['frequency'])           # Frequency (GHz)
        patch_len     = float(request.form['patch_length'])        # Patch length (mm)
        patch_wid     = float(request.form['patch_width'])         # Patch width (mm)
        substr_len    = float(request.form['substrate_length'])    # Substrate length (mm)
        substr_wid    = float(request.form['substrate_width'])     # Substrate width (mm)

        # 2. Prepare feature vector (must match training order)
        X = np.array([[freq, patch_len, patch_wid, substr_len, substr_wid]])

        # 3. Predict S11 (dB)
        s11_pred = model.predict(X)[0]

        # 4. Round and format result
        result = {
            's11_db': f"{s11_pred:.2f}"
        }

    # 5. Render the template with or without result
    return render_template('index.html', result=result)

if __name__ == '__main__':
    # Run on localhost:5000 by default
    app.run(debug=True)
