# 🍺 Beer Alcohol Predictor

Predict **total litres of pure alcohol** per capita using ML.

## Project Structure
```
beer_app/
├── app.py               # Flask application
├── train_model.py       # Model training script
├── flask_app.py         # WSGI entry for PythonAnywhere
├── requirements.txt
├── beer-servings.csv    # Dataset
├── model.pkl            # Trained model (generated)
├── scaler.pkl           # Feature scaler (generated)
├── le_country.pkl       # Country encoder (generated)
├── le_continent.pkl     # Continent encoder (generated)
├── model_meta.json      # Metrics & metadata (generated)
├── static/
│   └── chart_data.json  # Chart data (generated)
└── templates/
    └── index.html       # UI with infographics + prediction form
```

## Models Compared
| Model | R² | RMSE | MAE |
|---|---|---|---|
| Linear Regression | ~0.93 | ~1.07 | ~0.70 |
| Random Forest (GridSearchCV) | ~0.91 | ~1.22 | ~0.83 |

**Best model selected: Linear Regression** (highest R²)

## Run Locally
```bash
pip install -r requirements.txt
python train_model.py   # generates model files
python app.py           # starts Flask on localhost:5000
```

## Deploy to PythonAnywhere

1. **Create a free account** at https://www.pythonanywhere.com

2. **Upload files** via the Files tab (upload the entire `beer_app/` folder)

3. **Open a Bash console** and run:
   ```bash
   cd beer_app
   pip3 install --user -r requirements.txt
   python3 train_model.py
   ```

4. **Create a Web App**:
   - Go to Web tab → Add a new web app
   - Choose **Manual configuration** → Python 3.10
   - Set **Source code**: `/home/<username>/beer_app`
   - Set **Working directory**: `/home/<username>/beer_app`

5. **Edit the WSGI file** (link shown in the Web tab):
   - Replace the entire content with the contents of `flask_app.py`
   - Replace `<your-username>` with your actual PythonAnywhere username

6. **Reload** the web app — your app is live at:
   `https://<username>.pythonanywhere.com`

## Features
- 📊 Infographic landing page (bar, pie, scatter charts)
- 🔮 Prediction form with 3 numeric inputs + 2 dropdowns
- 🏆 Model comparison table (Linear Regression vs Random Forest)
- 📱 Responsive dark-theme UI
