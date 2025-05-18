import mlflow
import mlflow.sklearn
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris  # à remplacer par tes vraies données
import joblib
import pandas as pd
import matplotlib.pyplot as plt



# ➕ URI du tracking MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Charger les données
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modèle SVM
model = svm.SVC()

# ➕ Démarrer le tracking
with mlflow.start_run() as run:
    run_id = run.info.run_id
    model.fit(X_train_scaled, y_train)

    acc = model.score(X_test_scaled, y_test)

    # ➕ Prédictions
    y_pred = model.predict(X_test_scaled)

    # ➕ Afficher 10 exemples
    print("🧪 Exemples de prédictions :")
    for i in range(10):
        print(f"Réel: {y_test[i]} - Prédit: {y_pred[i]}")

    # ➕ Log des métriques
    mlflow.log_metric("accuracy", acc)

    # ➕ Enregistrer le modèle et le scaler
    mlflow.sklearn.log_model(model, "model")
    mlflow.sklearn.log_model(scaler, "scaler")

    # ➕ Enregistrer les prédictions dans un CSV
    pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    pred_df.to_csv("predictions.csv", index=False)
    mlflow.log_artifact("predictions.csv")

    # ➕ Sauvegarder un graphique
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Réel")
    plt.ylabel("Prédit")
    plt.title("Réel vs Prédit")
    plt.grid(True)
    plt.savefig("pred_vs_actual.png")
    mlflow.log_artifact("pred_vs_actual.png")

    print(f"✔️ Modèle et artefacts enregistrés (run ID: {run_id})")
