from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from texture_features import extract_texture_features

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
CLASSES = {"normal": 0, "defective": 1}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def get_images(folder):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and not p.stem.lower().endswith("_mask")
    )


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    X, y, paths = [], [], []

    for class_name, label in CLASSES.items():
        folder = DATASET / class_name
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")
        images = get_images(folder)
        print(f"{class_name}: found {len(images)} images")
        for image_path in images:
            try:
                X.append(extract_texture_features(str(image_path)))
                y.append(label)
                paths.append(str(image_path.relative_to(ROOT)))
            except Exception as exc:
                print(f"Skipping {image_path.name}: {exc}")

    if len(X) < 10:
        raise ValueError(
            "Not enough readable images. Add at least 10 images total "
            "and ensure both dataset/normal and dataset/defective have images."
        )
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if len(np.unique(y)) != 2:
        raise ValueError("Both normal and defective images are required.")
    if np.bincount(y, minlength=2).min() < 2:
        raise ValueError("Each class needs at least two readable images.")

    print(f"\nTotal usable images: {len(X)}")
    print(f"Feature vector length: {X.shape[1]}")
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.25, random_state=42, stratify=y
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = RandomForestClassifier(
        n_estimators=300, max_features="sqrt", min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
    report_text = classification_report(
        y_test, predictions, labels=[0, 1],
        target_names=["Normal", "Defective"], zero_division=0
    )
    report_dict = classification_report(
        y_test, predictions, labels=[0, 1],
        target_names=["Normal", "Defective"], output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1 score: {macro_f1:.4f}")
    print(report_text)
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(matrix)

    joblib.dump(model, MODEL_DIR / "fabric_model.joblib")
    metadata = {
        "classes": {"0": "Normal", "1": "Defective"},
        "feature_count": int(X.shape[1]),
        "train_images": int(len(train_idx)),
        "test_images": int(len(test_idx)),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classification_report": report_dict,
        "confusion_matrix": matrix.tolist(),
        "test_files": [paths[int(i)] for i in test_idx],
        "note": "Preliminary random image split; related fabric types may occur in both sets."
    }
    with open(REPORT_DIR / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nModel saved to: {MODEL_DIR / 'fabric_model.joblib'}")
    print(f"Report saved to: {REPORT_DIR / 'evaluation.json'}")


if __name__ == "__main__":
    main()
