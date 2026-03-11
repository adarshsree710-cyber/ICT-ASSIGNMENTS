from flask import Flask, render_template, request, jsonify
import pickle, json, numpy as np

app = Flask(__name__)

# ── Load artefacts ────────────────────────────────────────────────────────────
with open("model.pkl",       "rb") as f: model       = pickle.load(f)
with open("scaler.pkl",      "rb") as f: scaler      = pickle.load(f)
with open("le_country.pkl",  "rb") as f: le_country  = pickle.load(f)
with open("le_continent.pkl","rb") as f: le_continent= pickle.load(f)
with open("model_meta.json")       as f: meta        = json.load(f)
with open("static/chart_data.json")as f: chart_data  = json.load(f)

countries  = sorted(le_country.classes_.tolist())
continents = sorted(le_continent.classes_.tolist())

@app.route("/")
def index():
    return render_template(
        "index.html",
        countries=countries,
        continents=continents,
        meta=meta,
        chart_data=json.dumps(chart_data)
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        beer    = float(data["beer_servings"])
        spirit  = float(data["spirit_servings"])
        wine    = float(data["wine_servings"])
        country = data["country"]
        cont    = data["continent"]

        country_enc  = le_country.transform([country])[0]
        cont_enc     = le_continent.transform([cont])[0]

        X = np.array([[beer, spirit, wine, country_enc, cont_enc]])

        if meta["use_scaler"]:
            X = scaler.transform(X)

        prediction = model.predict(X)[0]
        prediction = round(max(0, prediction), 2)
        return jsonify({"prediction": prediction, "status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 400

if __name__ == "__main__":
    app.run(debug=True)
