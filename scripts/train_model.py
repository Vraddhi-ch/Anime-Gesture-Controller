import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# -----------------------------
# Paths
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

CSV_PATH = os.path.join(PROJECT_DIR, "dataset", "gestures.csv")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_classifier.joblib")

# -----------------------------
# Load dataset
# -----------------------------
if not os.path.exists(CSV_PATH):
    print("Dataset file not found:")
    print(CSV_PATH)
    raise SystemExit

data = pd.read_csv(CSV_PATH)

if data.empty:
    print("Your dataset is empty. Collect gesture samples first.")
    raise SystemExit

print("\nDataset shape:", data.shape)
print("\nSamples per gesture:")
print(data["label"].value_counts())

# -----------------------------
# Check minimum samples
# -----------------------------
class_counts = data["label"].value_counts()

if len(class_counts) < 2:
    print("\nYou need at least two gesture classes to train a classifier.")
    raise SystemExit

if class_counts.min() < 8:
    print("\nEach gesture needs at least 8 samples.")
    print("Collect more data, especially for the smallest class.")
    raise SystemExit

# -----------------------------
# Separate inputs and labels
# -----------------------------
X = data.drop(columns=["label"])
y = data["label"]

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# -----------------------------
# Train model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("\nTraining model...")
model.fit(X_train, y_train)

# -----------------------------
# Evaluate model
# -----------------------------
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n========== RESULTS ==========")
print(f"Test accuracy: {accuracy * 100:.2f}%")

print("\nClassification report:")
print(classification_report(y_test, predictions))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

# -----------------------------
# Save model and feature names
# -----------------------------
os.makedirs(MODEL_DIR, exist_ok=True)

saved_data = {
    "model": model,
    "feature_names": list(X.columns),
    "classes": list(model.classes_)
}

joblib.dump(saved_data, MODEL_PATH)

print("\nModel saved successfully at:")
print(MODEL_PATH)