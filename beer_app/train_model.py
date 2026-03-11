"""
Beer Alcohol Predictor — Model Training
Models: Linear Regression, Random Forest (+ hyperparameter tuning)
Target: total_litres_of_pure_alcohol
"""

import pandas as pd
import numpy as np
import pickle, json, os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ── 1. Load & clean ──────────────────────────────────────────────────────────
df = pd.read_csv("beer-servings.csv", index_col=0)
df = df.dropna(subset=["total_litres_of_pure_alcohol"])

# Fill numeric NaNs with median
for col in ["beer_servings", "spirit_servings", "wine_servings"]:
    df[col] = df[col].fillna(df[col].median())

# ── 2. Encode categoricals ───────────────────────────────────────────────────
le_country   = LabelEncoder()
le_continent = LabelEncoder()

df["country_enc"]   = le_country.fit_transform(df["country"])
df["continent_enc"] = le_continent.fit_transform(df["continent"])

features = ["beer_servings", "spirit_servings", "wine_servings",
            "country_enc", "continent_enc"]
target   = "total_litres_of_pure_alcohol"

X = df[features]
y = df[target]

# ── 3. Train / test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 4. Model 1 — Linear Regression ──────────────────────────────────────────
lr = LinearRegression()
lr.fit(X_train_sc, y_train)
lr_pred = lr.predict(X_test_sc)
lr_r2   = r2_score(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_mae  = mean_absolute_error(y_test, lr_pred)
print(f"Linear Regression  →  R²={lr_r2:.4f}  RMSE={lr_rmse:.4f}  MAE={lr_mae:.4f}")

# ── 5. Model 2 — Random Forest + GridSearchCV ────────────────────────────────
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth":    [None, 5, 10],
    "min_samples_split": [2, 5]
}
rf_base = RandomForestRegressor(random_state=42)
gs = GridSearchCV(rf_base, param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=0)
gs.fit(X_train, y_train)       # RF doesn't need scaling
best_rf = gs.best_estimator_
rf_pred = best_rf.predict(X_test)
rf_r2   = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mae  = mean_absolute_error(y_test, rf_pred)
print(f"Random Forest (CV) →  R²={rf_r2:.4f}  RMSE={rf_rmse:.4f}  MAE={rf_mae:.4f}")
print(f"Best RF params: {gs.best_params_}")

# ── 6. Choose best model ─────────────────────────────────────────────────────
if rf_r2 >= lr_r2:
    best_model      = best_rf
    best_model_name = "Random Forest"
    use_scaler      = False
    best_r2         = rf_r2
else:
    best_model      = lr
    best_model_name = "Linear Regression"
    use_scaler      = True
    best_r2         = lr_r2

print(f"\n✅ Best model: {best_model_name}  (R²={best_r2:.4f})")

# ── 7. Save artefacts ────────────────────────────────────────────────────────
with open("model.pkl",      "wb") as f: pickle.dump(best_model, f)
with open("scaler.pkl",     "wb") as f: pickle.dump(scaler, f)
with open("le_country.pkl", "wb") as f: pickle.dump(le_country, f)
with open("le_continent.pkl","wb") as f: pickle.dump(le_continent, f)

meta = {
    "best_model":      best_model_name,
    "use_scaler":      use_scaler,
    "r2_score":        round(best_r2, 4),
    "rmse":            round(float(rf_rmse if rf_r2>=lr_r2 else lr_rmse), 4),
    "mae":             round(float(rf_mae  if rf_r2>=lr_r2 else lr_mae),  4),
    "features":        features,
    "countries":       le_country.classes_.tolist(),
    "continents":      le_continent.classes_.tolist(),
    "model_comparison": {
        "Linear Regression": {"r2": round(lr_r2,4), "rmse": round(lr_rmse,4), "mae": round(lr_mae,4)},
        "Random Forest":     {"r2": round(rf_r2,4), "rmse": round(rf_rmse,4), "mae": round(rf_mae,4)},
    }
}
with open("model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("✅ Saved: model.pkl, scaler.pkl, le_country.pkl, le_continent.pkl, model_meta.json")

# ── 8. Generate infographic data ─────────────────────────────────────────────
df_clean = df.copy()
df_clean["country"]   = le_country.inverse_transform(df_clean["country_enc"])
df_clean["continent"] = le_continent.inverse_transform(df_clean["continent_enc"])

# Top 10 by beer servings
top_beer = df_clean.nlargest(10, "beer_servings")[["country","beer_servings"]].to_dict(orient="records")
# Continent avg alcohol
cont_avg = df_clean.groupby("continent")["total_litres_of_pure_alcohol"].mean().round(2).to_dict()
# Scatter data: beer_servings vs alcohol
scatter  = df_clean[["country","beer_servings","total_litres_of_pure_alcohol","continent"]].dropna().to_dict(orient="records")

chart_data = {
    "top_beer":  top_beer,
    "cont_avg":  cont_avg,
    "scatter":   scatter[:193]
}
with open("static/chart_data.json", "w") as f:
    json.dump(chart_data, f)

print("✅ chart_data.json written to static/")
