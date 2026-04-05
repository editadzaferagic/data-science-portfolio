import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

df = pd.read_csv("data/products.csv")
df.columns = df.columns.str.strip()

label_map = {
    "fridge": "Fridges",
    "CPU": "CPUs",
    "Mobile Phone": "Mobile Phones"
}

df["Category Label"] = df["Category Label"].replace(label_map)
df = df.dropna(subset=["Product Title", "Category Label"]).copy()
df["Product Title"] = df["Product Title"].astype(str).str.strip().str.lower()
df = df[df["Product Title"] != ""]

X = df["Product Title"]
y = df["Category Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "SGDClassifier": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", SGDClassifier(random_state=42, max_iter=1000, tol=1e-3))
    ]),
    "ComplementNB": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", ComplementNB(alpha=0.5))
    ])
}

best_model = None
best_model_name = None
best_score = 0
best_preds = None

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"{name}: {acc:.4f}")

    if acc > best_score:
        best_score = acc
        best_model = model
        best_model_name = name
        best_preds = preds

print(f"\nBest model: {best_model_name}")
print(f"Accuracy: {best_score:.4f}")

report = classification_report(y_test, best_preds)
print("\nClassification report:\n")
print(report)

with open("reports/classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

joblib.dump(best_model, "models/product_category_model.pkl")

cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(10, 8))
plt.imshow(cm)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png", dpi=200)

summary = {
    "best_model": best_model_name,
    "accuracy": round(best_score, 4)
}

with open("reports/metrics_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("Model and reports saved successfully.")