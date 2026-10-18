# Fabric Defect Detection Using Texture Analysis

A Python + Streamlit project that extracts GLCM, Local Binary Pattern (LBP), and Gabor texture features and uses a Random Forest classifier to predict **Normal** or **Defective** fabric.

## Important dataset note

This ZIP contains all source code and empty dataset folders, but **does not contain the AITEX dataset**. The dataset is hosted externally and must be downloaded separately; do not mistake generated/demo images for a valid training dataset. See https://www.aitex.es/afid/ for the official AITEX Fabric Image Database.

Organize real labeled images like this:

```text
dataset/
  normal/       # defect-free fabric images
  defective/    # fabric images containing defects
```

Do not place defect mask files (often named with `_mask`) in these folders. The training script skips filenames ending in `_mask`.

## Requirements

- Windows 10/11
- Python 3.11 or 3.12 recommended
- VS Code with the Python extension

## Setup in PowerShell

Open this project folder in VS Code, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If activation is blocked in the current PowerShell session:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

## Train and launch

After adding labeled images to both dataset folders:

```powershell
python train_model.py
python -m streamlit run app.py
```

Open the local URL Streamlit prints, usually http://localhost:8501.

## Outputs

- `models/fabric_model.joblib`: trained classifier
- `reports/evaluation.json`: test metrics and confusion matrix

## How it works

1. Convert input image to grayscale.
2. Calculate GLCM properties at multiple distances and angles.
3. Calculate an LBP histogram.
4. Calculate Gabor filter response statistics.
5. Train a Random Forest classifier.
6. Display image-level prediction and probability estimates in Streamlit.

## Limitations

- This first version predicts image-level labels; it does not locate defects or draw bounding boxes.
- You need real, correctly labeled examples from both classes before training.
- Random image-level splits can overestimate generalization when similar fabric types appear in both train and test sets. For research, split by fabric type or acquisition group.
- Class probabilities are not guarantees of correctness.
- This is an academic prototype, not a validated industrial inspection system.
