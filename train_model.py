# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# 1. Charger le dataset
df = pd.read_csv("electricity_cost_dataset.csv")

# 2. Renommer les colonnes mal orthographiées
df.rename(columns={
    "air qality index": "air quality index",
    "issue reolution time": "issue resolution time"
}, inplace=True)

# 3. Définir features et cible
target_column = "electricity cost"
X = df.drop(columns=[target_column])
y = df[target_column]

# 4. Définir colonnes numériques et catégorielles
categorical_features = ["structure type"]
numerical_features = X.drop(columns=categorical_features).columns.tolist()

# 5. Préparer un pipeline de transformation + modèle
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
], remainder="passthrough")

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

# 6. Séparation des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Entraînement
pipeline.fit(X_train, y_train)

# 8. Évaluation
y_pred = pipeline.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE : {rmse:.2f}")
print(f"R² : {r2:.2f}")

# 9. Sauvegarde
joblib.dump(pipeline, "model.joblib")
print("Modèle sauvegardé sous model.joblib")

